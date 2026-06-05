from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from .lora import LoRAConv2d
from .prompt_module import DegradationPromptModule


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MDTAAttention(nn.Module):
    """Restormer-style channel attention with optional LoRA on Q and V."""

    def __init__(self, channels: int, heads: int = 4, use_lora: bool = False, lora_rank: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q_lora = LoRAConv2d(channels, lora_rank) if use_lora else None
        self.v_lora = LoRAConv2d(channels, lora_rank) if use_lora else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.dwconv(self.qkv(x)).chunk(3, dim=1)
        if self.q_lora is not None:
            q = q + self.q_lora(x)
        if self.v_lora is not None:
            v = v + self.v_lora(x)

        q = rearrange(q, "b (head ch) h w -> b head ch (h w)", head=self.heads)
        k = rearrange(k, "b (head ch) h w -> b head ch (h w)", head=self.heads)
        v = rearrange(v, "b (head ch) h w -> b head ch (h w)", head=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b head ch (h w) -> b (head ch) h w", h=h, w=w)
        return self.project_out(out)


class GatedFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, use_lora: bool, lora_rank: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTAAttention(channels, heads=heads, use_lora=use_lora, lora_rank=lora_rank)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedFeedForward(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def _blocks(count: int, channels: int, heads: int, use_lora: bool, lora_rank: int) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(channels, heads, use_lora, lora_rank) for _ in range(count)))


class PromptLoRARestormer(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        blocks: tuple[int, int, int] = (2, 2, 3),
        heads: tuple[int, int, int] = (1, 2, 4),
        use_prompt: bool = True,
        use_lora: bool = True,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.use_prompt = use_prompt
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.encoder1 = _blocks(blocks[0], dim, heads[0], use_lora, lora_rank)
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.encoder2 = _blocks(blocks[1], dim * 2, heads[1], use_lora, lora_rank)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        self.bottleneck = _blocks(blocks[2], dim * 4, heads[2], use_lora, lora_rank)
        self.prompt = DegradationPromptModule(dim * 4) if use_prompt else None

        self.up2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, kernel_size=1), nn.PixelShuffle(2))
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder2 = _blocks(blocks[1], dim * 2, heads[1], use_lora, lora_rank)
        self.up1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, kernel_size=1), nn.PixelShuffle(2))
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder1 = _blocks(blocks[0], dim, heads[0], use_lora, lora_rank)

        self.refine = _blocks(1, dim, heads[0], use_lora, lora_rank)
        self.output = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        degradation: torch.Tensor | None = None,
        mix_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inp = x
        x1 = self.encoder1(self.patch_embed(x))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        if self.prompt is not None:
            if degradation is None or mix_weights is None:
                raise ValueError("degradation and mix_weights are required when prompts are enabled")
            x3 = self.prompt(x3, degradation, mix_weights)

        x = self.up2(x3)
        x = self.reduce2(torch.cat([x, x2], dim=1))
        x = self.decoder2(x)
        x = self.up1(x)
        x = self.reduce1(torch.cat([x, x1], dim=1))
        x = self.decoder1(x)
        x = self.refine(x)
        return (inp + self.output(x)).clamp(0.0, 1.0)


def build_model(variant: str = "prompt_lora", dim: int = 32, lora_rank: int = 4) -> PromptLoRARestormer:
    variants = {
        "baseline": (False, False),
        "prompt": (True, False),
        "lora": (False, True),
        "prompt_lora": (True, True),
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {sorted(variants)}")
    use_prompt, use_lora = variants[variant]
    return PromptLoRARestormer(dim=dim, use_prompt=use_prompt, use_lora=use_lora, lora_rank=lora_rank)


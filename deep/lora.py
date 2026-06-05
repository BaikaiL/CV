from __future__ import annotations

import math

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    """Low-rank residual adapter for 1x1 convolution projections."""

    def __init__(self, channels: int, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.scale = alpha / rank
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


def mark_only_lora_prompt_trainable(model: nn.Module) -> None:
    """Freeze the backbone and keep adapters, prompts, and output heads trainable."""

    trainable_keywords = ("lora", "prompt", "output", "refine")
    for name, param in model.named_parameters():
        param.requires_grad = any(keyword in name for keyword in trainable_keywords)


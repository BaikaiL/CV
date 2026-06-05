from __future__ import annotations

import torch
from torch import nn

from .dataset import BASE_DEGRADATIONS, DEGRADATION_TO_INDEX


class DegradationPromptModule(nn.Module):
    """Learnable degradation prompts injected as channel-wise feature biases."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.prompts = nn.Parameter(torch.zeros(len(BASE_DEGRADATIONS), channels))
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        nn.init.normal_(self.prompts, std=0.02)

    def forward(self, features: torch.Tensor, degradation: torch.Tensor, mix_weights: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        prompt = features.new_zeros(batch, self.channels)

        for idx, name in enumerate(BASE_DEGRADATIONS):
            mask = degradation == DEGRADATION_TO_INDEX[name]
            if mask.any():
                prompt[mask] = self.prompts[idx]

        mixed_mask = degradation == DEGRADATION_TO_INDEX["mixed"]
        if mixed_mask.any():
            weights = mix_weights[mixed_mask].to(features.device, features.dtype)
            prompt[mixed_mask] = weights @ self.prompts.to(features.dtype)

        prompt = self.proj(prompt).view(batch, self.channels, 1, 1)
        return features + prompt


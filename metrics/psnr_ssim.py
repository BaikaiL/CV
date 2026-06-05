from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_psnr(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = F.mse_loss(output.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def calculate_ssim(output: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    x = output.clamp(0, 1)
    y = target.clamp(0, 1)
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(dim=1)


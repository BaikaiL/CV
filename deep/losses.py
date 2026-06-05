from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
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
    return score.clamp(0.0, 1.0).mean()


class SSIMLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - _ssim(output, target)


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kx = self.kernel_x.repeat(channels, 1, 1, 1)
        ky = self.kernel_y.repeat(channels, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=channels)
        gy = F.conv2d(x, ky, padding=1, groups=channels)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(output), self._edges(target))


class RestorationLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.1, edge_weight: float = 0.05) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = F.l1_loss(output, target)
        ssim = self.ssim(output, target)
        edge = self.edge(output, target)
        total = l1 + self.ssim_weight * ssim + self.edge_weight * edge
        return total, {"l1": l1.item(), "ssim_loss": ssim.item(), "edge": edge.item()}


"""Synthetic haze formation based on the atmospheric scattering model."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _smooth_depth_proxy(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Build a smooth pseudo-depth map when real depth is unavailable."""

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    vertical = 1.0 - y
    radial = np.sqrt((x - 0.5) ** 2 + (y - 0.52) ** 2)
    radial = radial / max(float(radial.max()), 1e-6)

    coarse_h = max(4, height // 96)
    coarse_w = max(4, width // 96)
    noise = rng.random((coarse_h, coarse_w), dtype=np.float32)
    noise_img = Image.fromarray((noise * 255.0).astype(np.uint8), mode="L")
    noise_img = noise_img.resize((width, height), resample=Image.Resampling.BICUBIC)
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 28.0))
    noise = np.asarray(noise_img, dtype=np.float32) / 255.0
    noise = (noise - noise.min()) / max(float(noise.max() - noise.min()), 1e-6)

    depth = 0.60 * vertical + 0.25 * radial + 0.15 * noise
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def add_haze(
    image: np.ndarray,
    beta: float = 1.8,
    airlight: tuple[float, float, float] = (0.86, 0.88, 0.90),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply I(x) = J(x)t(x) + A(1 - t(x)) to an RGB uint8 image."""

    rng = rng or np.random.default_rng()
    clean = image.astype(np.float32) / 255.0
    height, width = clean.shape[:2]
    depth = _smooth_depth_proxy(height, width, rng)
    transmission = np.exp(-beta * depth)[..., None]
    atmospheric_light = np.asarray(airlight, dtype=np.float32).reshape(1, 1, 3)

    hazy = clean * transmission + atmospheric_light * (1.0 - transmission)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)

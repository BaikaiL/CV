"""Low-light synthesis for clean RGB images."""

from __future__ import annotations

import numpy as np


def add_lowlight(
    image: np.ndarray,
    gamma: float = 2.2,
    gain: float = 0.58,
    noise_std: float = 0.012,
    color_shift: tuple[float, float, float] = (1.02, 0.96, 0.90),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Darken an RGB uint8 image with gamma, attenuation, color cast, and shot noise.

    This creates the LQ low-light image. It intentionally uses gamma > 1 because
    the degradation step should darken the clean image; restoration later can use
    gamma < 1 to brighten it.
    """

    rng = rng or np.random.default_rng()
    x = image.astype(np.float32) / 255.0

    degraded = np.power(np.clip(x * gain, 0.0, 1.0), gamma)
    degraded *= np.asarray(color_shift, dtype=np.float32).reshape(1, 1, 3)

    rgb = np.clip(degraded, 0.0, 1.0)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    signal_noise = rng.normal(0.0, noise_std, size=degraded.shape).astype(np.float32)
    shot_scale = np.sqrt(np.clip(luminance, 0.02, 1.0))[..., None]
    degraded = degraded + signal_noise * shot_scale

    return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)

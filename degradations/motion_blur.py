"""Motion blur synthesis for clean RGB images."""

from __future__ import annotations

import numpy as np


def motion_blur_kernel(length: int = 21, angle: float = 0.0) -> np.ndarray:
    """Create a normalized 2D linear motion blur kernel."""

    length = max(3, int(length))
    if length % 2 == 0:
        length += 1

    center = (length - 1) / 2.0
    yy, xx = np.mgrid[0:length, 0:length].astype(np.float32)
    xx -= center
    yy -= center

    theta = np.deg2rad(angle)
    along = xx * np.cos(theta) + yy * np.sin(theta)
    across = -xx * np.sin(theta) + yy * np.cos(theta)
    kernel = (np.abs(along) <= center).astype(np.float32) * np.exp(-(across**2) / 0.5)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        kernel[center, center] = 1.0
        kernel_sum = 1.0
    return kernel / kernel_sum


def _fft_convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    kh, kw = kernel.shape
    fft_shape = (height + kh - 1, width + kw - 1)
    kernel_fft = np.fft.rfft2(kernel, fft_shape)

    output = np.empty_like(image, dtype=np.float32)
    top = kh // 2
    left = kw // 2
    for channel in range(image.shape[2]):
        image_fft = np.fft.rfft2(image[..., channel], fft_shape)
        convolved = np.fft.irfft2(image_fft * kernel_fft, fft_shape)
        output[..., channel] = convolved[top : top + height, left : left + width]
    return output


def add_motion_blur(
    image: np.ndarray,
    length: int = 21,
    angle: float = 0.0,
    noise_std: float = 0.004,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply linear motion blur plus light sensor noise to an RGB uint8 image."""

    rng = rng or np.random.default_rng()
    kernel = motion_blur_kernel(length=length, angle=angle)
    x = image.astype(np.float32) / 255.0
    pad = length // 2
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    blurred = _fft_convolve_rgb(padded, kernel)[pad:-pad, pad:-pad, :]
    blurred += rng.normal(0.0, noise_std, size=blurred.shape).astype(np.float32)
    return np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

"""JPEG compression artifact synthesis for clean RGB images."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def add_jpeg_artifacts(image: np.ndarray, quality: int = 25) -> np.ndarray:
    """Round-trip an RGB uint8 image through JPEG encoding at the given quality."""

    quality = int(np.clip(quality, 5, 95))
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)

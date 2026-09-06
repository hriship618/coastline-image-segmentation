"""Small synthetic examples used to define and test the data contract."""

from __future__ import annotations

import numpy as np

LAND_CLASS = 0
WATER_CLASS = 1


def make_synthetic_coast(
    height: int = 128,
    width: int = 128,
    channels: int = 5,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fake multispectral coastal image and its land/water mask.

    The curving boundary is intentionally simple. Values on the left represent
    land and values on the right represent water. The spectral values differ by
    class so a future model can learn a real pixel-to-mask relationship during
    smoke tests.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if channels < 3:
        raise ValueError("channels must be at least 3")

    rng = np.random.default_rng(seed)
    rows = np.arange(height, dtype=np.float32)
    shoreline_x = width * 0.52 + width * 0.08 * np.sin(rows / height * 2 * np.pi)
    columns = np.arange(width, dtype=np.float32)[None, :]
    mask = np.where(columns >= shoreline_x[:, None], WATER_CLASS, LAND_CLASS)
    mask = mask.astype(np.int64)

    land_signature = np.linspace(0.45, 0.75, channels, dtype=np.float32)
    water_signature = np.linspace(0.35, 0.08, channels, dtype=np.float32)
    image = np.empty((channels, height, width), dtype=np.float32)

    for channel in range(channels):
        image[channel] = np.where(
            mask == WATER_CLASS,
            water_signature[channel],
            land_signature[channel],
        )

    noise = rng.normal(0.0, 0.025, size=image.shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)
    return image, mask


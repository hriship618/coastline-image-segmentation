"""SWED discovery and loading utilities.

The functions in this module deliberately keep file discovery separate from
pixel decoding. That lets us test naming and pairing locally without having to
download SWED or install the geospatial/ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Sequence

import numpy as np

LAND_CLASS = 0
WATER_CLASS = 1
IGNORE_INDEX = 255

# One-based positions in SWED's 12-band Sentinel-2 GeoTIFFs.
RGB_BANDS = (4, 3, 2)
FIVE_BANDS = (4, 3, 2, 8, 11)


@dataclass(frozen=True)
class ImageMaskPair:
    """Paths belonging to one supervised segmentation example."""

    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class GeographicSplits:
    """Non-overlapping train, validation, and test pairs grouped by region."""

    train: tuple[ImageMaskPair, ...]
    validation: tuple[ImageMaskPair, ...]
    test: tuple[ImageMaskPair, ...]


_SENTINEL_TILE_PATTERN = re.compile(r"_T(?P<tile>\d{2}[A-Z]{3})_")


def swed_region_id(pair: ImageMaskPair) -> str:
    """Extract the Sentinel-2 MGRS tile, used as a geographic group ID."""
    match = _SENTINEL_TILE_PATTERN.search(pair.image_path.name)
    if match is None:
        raise ValueError(
            "Could not extract a Sentinel tile such as T48QYJ from "
            f"{pair.image_path.name}"
        )
    return match.group("tile")


def split_pairs_by_region(
    pairs: Sequence[ImageMaskPair],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> GeographicSplits:
    """Assign entire Sentinel tiles to one split to prevent spatial leakage."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    regions = sorted({swed_region_id(pair) for pair in pairs})
    if len(regions) < 3:
        raise ValueError("Geographic splitting requires at least three Sentinel tiles")

    random.Random(seed).shuffle(regions)
    validation_count = max(1, round(len(regions) * validation_fraction))
    test_count = max(1, round(len(regions) * test_fraction))
    while validation_count + test_count >= len(regions):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Not enough regions to create three non-empty splits")

    test_regions = set(regions[:test_count])
    validation_regions = set(regions[test_count : test_count + validation_count])
    train_regions = set(regions) - validation_regions - test_regions

    def select(selected_regions: set[str]) -> tuple[ImageMaskPair, ...]:
        return tuple(pair for pair in pairs if swed_region_id(pair) in selected_regions)

    return GeographicSplits(
        train=select(train_regions),
        validation=select(validation_regions),
        test=select(test_regions),
    )


def discover_swed_pairs(root: str | Path) -> list[ImageMaskPair]:
    """Find SWED image files and require one matching label for each image."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SWED root does not exist: {root}")

    image_paths = sorted(root.rglob("*_image_*.tif"))
    pairs: list[ImageMaskPair] = []
    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_name = image_path.name.replace("_image_", "_label_", 1)
        same_directory_label = image_path.with_name(label_name)

        if same_directory_label.exists():
            label_path = same_directory_label
        else:
            matches = list(root.rglob(label_name))
            if len(matches) != 1:
                missing_labels.append(image_path)
                continue
            label_path = matches[0]

        pairs.append(ImageMaskPair(image_path=image_path, mask_path=label_path))

    if missing_labels:
        preview = ", ".join(path.name for path in missing_labels[:3])
        raise ValueError(
            f"Could not find exactly one label for {len(missing_labels)} image(s): {preview}"
        )
    if not pairs:
        raise ValueError(f"No SWED image/label pairs found below {root}")
    return pairs


def validate_band_positions(bands: Sequence[int], available_bands: int) -> tuple[int, ...]:
    """Validate the one-based band positions passed to rasterio."""
    selected = tuple(int(band) for band in bands)
    if not selected:
        raise ValueError("At least one image band must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Band positions must be unique: {selected}")
    if min(selected) < 1 or max(selected) > available_bands:
        raise ValueError(
            f"Band positions {selected} exceed raster's 1..{available_bands} range"
        )
    return selected


def read_swed_image(image_path: str | Path, bands: Sequence[int] = FIVE_BANDS) -> np.ndarray:
    """Read selected Sentinel-2 bands as normalized [C,H,W] float32 data."""
    import rasterio

    with rasterio.open(image_path) as source:
        selected = validate_band_positions(bands, source.count)
        image = source.read(selected).astype(np.float32)

    # Sentinel-2 L2A surface reflectance is conventionally scaled by 10,000.
    # Clipping also makes rare negative/outlier values safe for the first model.
    return np.clip(image / 10_000.0, 0.0, 1.0)


def read_swed_mask(mask_path: str | Path) -> np.ndarray:
    """Read a SWED binary mask as [H,W] int64, ignoring unexpected values."""
    import rasterio

    with rasterio.open(mask_path) as source:
        mask = source.read(1)

    normalized = np.full(mask.shape, IGNORE_INDEX, dtype=np.int64)
    normalized[mask == LAND_CLASS] = LAND_CLASS
    normalized[mask == WATER_CLASS] = WATER_CLASS
    return normalized


class SwedDataset:
    """PyTorch-compatible dataset returning an image, mask, and source paths."""

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        bands: Sequence[int] = FIVE_BANDS,
    ) -> None:
        if not pairs:
            raise ValueError("SwedDataset requires at least one image/mask pair")
        self.pairs = list(pairs)
        self.bands = tuple(bands)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        import torch

        pair = self.pairs[index]
        image = read_swed_image(pair.image_path, self.bands)
        mask = read_swed_mask(pair.mask_path)
        if image.shape[-2:] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: {image.shape[-2:]} versus {mask.shape}"
            )
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }


def build_dataloaders(
    splits: GeographicSplits,
    bands: Sequence[int] = FIVE_BANDS,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """Build loaders while shuffling only the training split."""
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    return {
        "train": DataLoader(
            SwedDataset(splits.train, bands=bands), shuffle=True, **loader_options
        ),
        "validation": DataLoader(
            SwedDataset(splits.validation, bands=bands), shuffle=False, **loader_options
        ),
        "test": DataLoader(
            SwedDataset(splits.test, bands=bands), shuffle=False, **loader_options
        ),
    }

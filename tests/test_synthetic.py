"""Tests for the synthetic shoreline data contract."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.synthetic import LAND_CLASS, WATER_CLASS, make_synthetic_coast


class SyntheticCoastTests(unittest.TestCase):
    def test_shapes_and_types(self) -> None:
        image, mask = make_synthetic_coast(height=64, width=96, channels=5)

        self.assertEqual(image.shape, (5, 64, 96))
        self.assertEqual(mask.shape, (64, 96))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(mask.dtype, np.int64)

    def test_mask_contains_land_and_water(self) -> None:
        _, mask = make_synthetic_coast()

        self.assertEqual(set(np.unique(mask)), {LAND_CLASS, WATER_CLASS})

    def test_seed_is_reproducible(self) -> None:
        first_image, first_mask = make_synthetic_coast(seed=11)
        second_image, second_mask = make_synthetic_coast(seed=11)

        np.testing.assert_array_equal(first_image, second_image)
        np.testing.assert_array_equal(first_mask, second_mask)


if __name__ == "__main__":
    unittest.main()


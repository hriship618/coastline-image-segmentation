"""Dependency-light tests for web input preprocessing."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.webapp import prepare_satellite_array


class WebInputTests(unittest.TestCase):
    def test_selects_five_bands_from_channel_last_array(self) -> None:
        image = np.zeros((32, 40, 12), dtype=np.uint16)
        for channel in range(12):
            image[..., channel] = (channel + 1) * 100

        prepared = prepare_satellite_array(image)

        self.assertEqual(prepared.shape, (5, 32, 40))
        np.testing.assert_allclose(
            prepared[:, 0, 0], np.array([0.04, 0.03, 0.02, 0.08, 0.11])
        )

    def test_accepts_prepared_channel_first_array(self) -> None:
        image = np.full((5, 32, 32), 0.25, dtype=np.float32)
        prepared = prepare_satellite_array(image)
        np.testing.assert_array_equal(prepared, image)

    def test_rejects_rgb_only_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "require"):
            prepare_satellite_array(np.zeros((32, 32, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

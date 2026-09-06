"""Shape validation tests that do not require PyTorch."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.training import validate_segmentation_shapes


class TrainingShapeTests(unittest.TestCase):
    def test_valid_shapes(self) -> None:
        logits = np.zeros((2, 2, 64, 64))
        masks = np.zeros((2, 64, 64))
        validate_segmentation_shapes(logits, masks)

    def test_spatial_mismatch_is_rejected(self) -> None:
        logits = np.zeros((2, 2, 32, 32))
        masks = np.zeros((2, 64, 64))
        with self.assertRaisesRegex(ValueError, "spatial sizes"):
            validate_segmentation_shapes(logits, masks)


if __name__ == "__main__":
    unittest.main()


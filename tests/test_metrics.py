"""Exact examples for understanding intersection-over-union."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.metrics import SegmentationConfusionMatrix


class SegmentationMetricTests(unittest.TestCase):
    def test_confusion_matrix_and_iou(self) -> None:
        targets = np.array([[0, 0], [1, 1]])
        predictions = np.array([[0, 1], [1, 1]])
        metric = SegmentationConfusionMatrix(num_classes=2)

        metric.update(predictions, targets)

        np.testing.assert_array_equal(metric.matrix, [[1, 1], [0, 2]])
        np.testing.assert_allclose(metric.per_class_iou(), [1 / 2, 2 / 3])
        self.assertAlmostEqual(metric.summary()["mean_iou"], 7 / 12)

    def test_ignored_pixels_do_not_affect_counts(self) -> None:
        targets = np.array([0, 1, 255])
        predictions = np.array([0, 1, 0])
        metric = SegmentationConfusionMatrix(num_classes=2, ignore_index=255)

        metric.update(predictions, targets)

        np.testing.assert_array_equal(metric.matrix, [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()


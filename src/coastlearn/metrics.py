"""Transparent segmentation metrics derived from a confusion matrix."""

from __future__ import annotations

import numpy as np


class SegmentationConfusionMatrix:
    """Accumulate rows=true classes and columns=predicted classes."""

    def __init__(self, num_classes: int = 2, ignore_index: int = 255) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions, targets) -> None:
        """Add one batch after removing ignored and invalid pixels."""
        predictions = np.asarray(predictions).reshape(-1)
        targets = np.asarray(targets).reshape(-1)
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must contain the same pixels")

        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
            & (predictions >= 0)
            & (predictions < self.num_classes)
        )
        encoded = self.num_classes * targets[valid] + predictions[valid]
        counts = np.bincount(encoded, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        """Return intersection-over-union for every class."""
        intersection = np.diag(self.matrix).astype(np.float64)
        target_pixels = self.matrix.sum(axis=1)
        predicted_pixels = self.matrix.sum(axis=0)
        union = target_pixels + predicted_pixels - intersection
        return np.divide(
            intersection,
            union,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=union != 0,
        )

    def summary(self) -> dict[str, object]:
        class_iou = self.per_class_iou()
        return {
            "class_iou": class_iou.tolist(),
            "mean_iou": float(np.nanmean(class_iou)),
            "confusion_matrix": self.matrix.tolist(),
        }


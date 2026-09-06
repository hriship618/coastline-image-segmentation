"""Dependency-light validation tests for model configuration behavior."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.models import (
    build_convnext_tiny,
    build_dinov3_convnext_tiny,
    build_model,
    build_resnet34_unet,
)


class ModelConfigurationTests(unittest.TestCase):
    def test_invalid_channel_count_fails_before_ml_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_channels"):
            build_resnet34_unet(in_channels=0)

    def test_invalid_class_count_fails_before_ml_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_classes"):
            build_resnet34_unet(num_classes=1)

    def test_unknown_model_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            build_model({"name": "mystery_network"})

    def test_invalid_convnext_inputs_fail_before_ml_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_channels"):
            build_convnext_tiny(in_channels=0)
        with self.assertRaisesRegex(ValueError, "num_classes"):
            build_convnext_tiny(num_classes=1)

    def test_invalid_dinov3_inputs_fail_before_ml_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_channels"):
            build_dinov3_convnext_tiny(in_channels=0)
        with self.assertRaisesRegex(ValueError, "num_classes"):
            build_dinov3_convnext_tiny(num_classes=1)


if __name__ == "__main__":
    unittest.main()

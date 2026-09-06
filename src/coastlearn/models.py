"""Segmentation model construction.

Imports of large ML libraries stay inside builder functions. This keeps simple
data-contract tests runnable on machines that do not have PyTorch installed.
"""

from __future__ import annotations

import os


def build_resnet34_unet(
    in_channels: int = 5,
    num_classes: int = 2,
    pretrained: bool = True,
):
    """Build a ResNet34 encoder with a UNet decoder.

    The encoder compresses the image into increasingly semantic feature maps.
    The decoder restores spatial resolution, using skip connections from the
    encoder to retain fine boundary information.
    """
    if in_channels < 1:
        raise ValueError("in_channels must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    from coastlearn.torch_models import TimmUNetSegmentationModel

    return TimmUNetSegmentationModel(
        backbone_name="resnet34",
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def build_convnext_tiny(
    in_channels: int = 5,
    num_classes: int = 2,
    pretrained: bool = True,
):
    """Build ConvNeXt-Tiny with the shared U-Net decoder."""
    if in_channels < 1:
        raise ValueError("in_channels must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    from coastlearn.torch_models import ConvNeXtSegmentationModel

    return ConvNeXtSegmentationModel(
        backbone_name="convnext_tiny",
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def build_dinov3_convnext_tiny(
    in_channels: int = 5,
    num_classes: int = 2,
    model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    token: str | None = None,
):
    """Build a DINOv3-pretrained ConvNeXt-Tiny segmentation model."""
    if in_channels < 1:
        raise ValueError("in_channels must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    from coastlearn.torch_models import DinoV3ConvNeXtSegmentationModel

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    return DinoV3ConvNeXtSegmentationModel(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        token=resolved_token,
    )


def build_model(model_config: dict):
    """Build the model described by a parsed YAML `model` section."""
    model_name = model_config.get("name")
    if model_name == "resnet34_unet":
        return build_resnet34_unet(
            in_channels=int(model_config.get("in_channels", 5)),
            num_classes=int(model_config.get("num_classes", 2)),
            pretrained=bool(model_config.get("pretrained", True)),
        )
    if model_name == "convnext_tiny":
        return build_convnext_tiny(
            in_channels=int(model_config.get("in_channels", 5)),
            num_classes=int(model_config.get("num_classes", 2)),
            pretrained=bool(model_config.get("pretrained", True)),
        )
    if model_name == "dinov3_convnext_tiny":
        return build_dinov3_convnext_tiny(
            in_channels=int(model_config.get("in_channels", 5)),
            num_classes=int(model_config.get("num_classes", 2)),
            model_name=model_config.get(
                "model_name", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
            ),
        )
    raise ValueError(f"Unsupported model name: {model_name!r}")

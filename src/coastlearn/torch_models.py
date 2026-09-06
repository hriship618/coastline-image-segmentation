"""PyTorch model classes imported only in ML-capable environments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class ConvNeXtSegmentationModel(nn.Module):
    """ConvNeXt feature extractor followed by a minimal segmentation head."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        in_channels: int = 5,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
        )

        deepest_channels = self.backbone.feature_info.channels()[-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(deepest_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return the full feature pyramid for inspection or later extensions."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        feature_pyramid = self.extract_features(images)
        deepest_features = feature_pyramid[-1]
        low_resolution_logits = self.decoder(deepest_features)
        return F.interpolate(
            low_resolution_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


class DinoV3ConvNeXtSegmentationModel(nn.Module):
    """DINOv3-pretrained ConvNeXt backbone plus a minimal segmentation head."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        in_channels: int = 5,
        num_classes: int = 2,
        token: str | None = None,
    ) -> None:
        super().__init__()
        hugging_face_model = AutoModel.from_pretrained(model_name, token=token)

        # Transformers versions may wrap the actual ConvNeXt module in `.model`.
        self.backbone = getattr(hugging_face_model, "model", hugging_face_model)
        self._adapt_input_stem(in_channels)

        deepest_channels = int(self.backbone.config.hidden_sizes[-1])
        self.decoder = nn.Sequential(
            nn.Conv2d(deepest_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def _adapt_input_stem(self, in_channels: int) -> None:
        """Replace the RGB stem while preserving its useful pretrained weights."""
        old_stem = self.backbone.stages[0].downsample_layers[0]
        if in_channels == old_stem.in_channels:
            return

        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            dilation=old_stem.dilation,
            groups=old_stem.groups,
            bias=old_stem.bias is not None,
            padding_mode=old_stem.padding_mode,
        ).to(device=old_stem.weight.device, dtype=old_stem.weight.dtype)

        with torch.no_grad():
            copied_channels = min(in_channels, old_stem.in_channels)
            new_stem.weight[:, :copied_channels].copy_(
                old_stem.weight[:, :copied_channels]
            )

            if in_channels > old_stem.in_channels:
                mean_rgb_weight = old_stem.weight.mean(dim=1, keepdim=True)
                extra_channel_count = in_channels - old_stem.in_channels
                new_stem.weight[:, old_stem.in_channels :].copy_(
                    mean_rgb_weight.repeat(1, extra_channel_count, 1, 1)
                )

            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        self.backbone.stages[0].downsample_layers[0] = new_stem

    @property
    def input_stem(self) -> nn.Conv2d:
        """Expose the adapted stem so its weights can be inspected in notebooks."""
        return self.backbone.stages[0].downsample_layers[0]

    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Run each backbone stage and return its output for inspection."""
        features = images
        feature_pyramid = []
        for stage in self.backbone.stages:
            features = stage(features)
            feature_pyramid.append(features)
        return feature_pyramid

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        deepest_features = self.extract_features(images)[-1]
        low_resolution_logits = self.decoder(deepest_features)
        return F.interpolate(
            low_resolution_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

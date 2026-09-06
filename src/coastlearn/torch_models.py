"""PyTorch model classes imported only in ML-capable environments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class ConvolutionBlock(nn.Sequential):
    """Two convolutions used after each decoder skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SharedUNetDecoder(nn.Module):
    """One decoder design shared by every encoder in the comparison."""

    def __init__(
        self,
        feature_channels: list[int] | tuple[int, ...],
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError("SharedUNetDecoder requires four feature levels")

        self.deepest = ConvolutionBlock(feature_channels[-1], decoder_channels[0])
        blocks = []
        current_channels = decoder_channels[0]
        for skip_channels, output_channels in zip(
            reversed(feature_channels[:-1]), decoder_channels[1:]
        ):
            blocks.append(
                ConvolutionBlock(current_channels + skip_channels, output_channels)
            )
            current_channels = output_channels
        self.skip_blocks = nn.ModuleList(blocks)
        self.refine = ConvolutionBlock(current_channels, current_channels)
        self.classifier = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(
        self, feature_pyramid: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        if len(feature_pyramid) != 4:
            raise ValueError("Expected four encoder feature maps")

        decoded = self.deepest(feature_pyramid[-1])
        for skip, block in zip(reversed(feature_pyramid[:-1]), self.skip_blocks):
            decoded = F.interpolate(
                decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = block(torch.cat([decoded, skip], dim=1))

        decoded = F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )
        return self.classifier(self.refine(decoded))


class TimmUNetSegmentationModel(nn.Module):
    """A timm feature encoder connected to the shared U-Net decoder."""

    def __init__(
        self,
        backbone_name: str,
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
        all_channels = list(self.backbone.feature_info.channels())
        if len(all_channels) < 4:
            raise ValueError(f"{backbone_name} does not expose four feature levels")
        self.feature_indices = tuple(range(len(all_channels) - 4, len(all_channels)))
        feature_channels = [all_channels[index] for index in self.feature_indices]
        self.decoder = SharedUNetDecoder(feature_channels, num_classes=num_classes)

    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(images)
        return [features[index] for index in self.feature_indices]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.extract_features(images), images.shape[-2:])


class ConvNeXtSegmentationModel(TimmUNetSegmentationModel):
    """ConvNeXt-Tiny connected to the shared U-Net decoder."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        in_channels: int = 5,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__(backbone_name, in_channels, num_classes, pretrained)


class DinoV3ConvNeXtSegmentationModel(nn.Module):
    """DINOv3-pretrained ConvNeXt-Tiny plus the shared U-Net decoder."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        in_channels: int = 5,
        num_classes: int = 2,
        token: str | None = None,
    ) -> None:
        super().__init__()
        hugging_face_model = AutoModel.from_pretrained(model_name, token=token)
        self.backbone = getattr(hugging_face_model, "model", hugging_face_model)
        self._adapt_input_stem(in_channels)

        feature_channels = [int(value) for value in self.backbone.config.hidden_sizes]
        self.decoder = SharedUNetDecoder(feature_channels, num_classes=num_classes)

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
        return self.backbone.stages[0].downsample_layers[0]

    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = images
        feature_pyramid = []
        for stage in self.backbone.stages:
            features = stage(features)
            feature_pyramid.append(features)
        return feature_pyramid

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.extract_features(images), images.shape[-2:])

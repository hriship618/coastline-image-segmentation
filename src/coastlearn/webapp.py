"""Minimal web interface for comparing the three shoreline models."""

from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
from threading import Lock
import numpy as np

from coastlearn.data import FIVE_BANDS, validate_band_positions


MODEL_SPECS = (
    (
        "ResNet-34",
        "COASTLEARN_RESNET_CHECKPOINT",
        "resnet34_unet",
    ),
    (
        "ConvNeXt-Tiny",
        "COASTLEARN_CONVNEXT_CHECKPOINT",
        "convnext_tiny",
    ),
    (
        "DINOv3 ConvNeXt-Tiny",
        "COASTLEARN_DINO_CHECKPOINT",
        "dinov3_convnext_tiny",
    ),
)


def prepare_satellite_array(array: np.ndarray) -> np.ndarray:
    """Convert an uploaded multispectral array to normalized ``[5,H,W]``."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D satellite array, received {array.shape}.")

    if array.shape[0] <= 16 and array.shape[1] > 16 and array.shape[2] > 16:
        channel_first = array
    elif array.shape[-1] <= 16 and array.shape[0] > 16 and array.shape[1] > 16:
        channel_first = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            "Could not identify the band dimension. Use [bands, height, width] "
            "or [height, width, bands]."
        )

    available_bands = channel_first.shape[0]
    if available_bands == 5:
        selected = channel_first
    elif available_bands >= max(FIVE_BANDS):
        positions = validate_band_positions(FIVE_BANDS, available_bands)
        selected = channel_first[[position - 1 for position in positions]]
    else:
        raise ValueError(
            "The models require either five prepared bands (RGB, NIR, SWIR) "
            "or a Sentinel-style array with at least 11 bands."
        )

    selected = selected.astype(np.float32)
    finite_values = selected[np.isfinite(selected)]
    if finite_values.size == 0:
        raise ValueError("The uploaded image contains no finite pixel values.")
    if float(np.percentile(finite_values, 99)) > 2.0:
        selected /= 10_000.0
    selected = np.nan_to_num(selected, nan=0.0, posinf=1.0, neginf=0.0)
    selected = np.clip(selected, 0.0, 1.0)

    height, width = selected.shape[-2:]
    if height < 32 or width < 32:
        raise ValueError("Images must be at least 32 × 32 pixels.")
    if height * width > 16_777_216:
        raise ValueError("Images may contain at most 16 million pixels.")
    return np.ascontiguousarray(selected)


def read_uploaded_satellite(file_storage) -> np.ndarray:
    """Read a Werkzeug upload as GeoTIFF or NumPy and normalize its bands."""
    filename = Path(file_storage.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        array = np.load(file_storage.stream, allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        from rasterio.io import MemoryFile

        payload = file_storage.read()
        with MemoryFile(payload) as memory_file:
            with memory_file.open() as dataset:
                array = dataset.read()
    else:
        raise ValueError("Upload a .tif, .tiff, or .npy multispectral image.")
    return prepare_satellite_array(array)


def _png_data_url(rgb: np.ndarray) -> str:
    from PIL import Image

    output = BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def rgb_preview_data_url(image: np.ndarray) -> str:
    """Render the first three normalized bands with simple contrast stretching."""
    rgb = np.moveaxis(image[:3], 0, -1)
    low, high = np.percentile(rgb, [2, 98])
    scale = max(float(high - low), 1e-6)
    rgb = np.clip((rgb - low) / scale, 0.0, 1.0)
    return _png_data_url(np.rint(rgb * 255.0))


def mask_data_url(mask: np.ndarray) -> str:
    """Render land as sand and water as blue, with a white shoreline."""
    mask = np.asarray(mask, dtype=np.uint8)
    colors = np.empty((*mask.shape, 3), dtype=np.uint8)
    colors[mask == 0] = (194, 166, 112)
    colors[mask == 1] = (45, 118, 180)

    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    colors[boundary] = (255, 255, 255)
    return _png_data_url(colors)


class SegmentationService:
    """Lazily load checkpoints and run the same image through each model."""

    def __init__(self, checkpoint_paths: dict[str, str] | None = None) -> None:
        self.checkpoint_paths = checkpoint_paths or {
            key: os.environ.get(environment_name, "")
            for _, environment_name, key in MODEL_SPECS
        }
        self._models = None
        self._device = None
        self._lock = Lock()

    def _load_models(self):
        import torch

        from coastlearn.models import (
            build_convnext_tiny,
            build_dinov3_convnext_tiny,
            build_resnet34_unet,
        )

        builders = {
            "resnet34_unet": lambda: build_resnet34_unet(
                in_channels=5, num_classes=2, pretrained=False
            ),
            "convnext_tiny": lambda: build_convnext_tiny(
                in_channels=5, num_classes=2, pretrained=False
            ),
            "dinov3_convnext_tiny": lambda: build_dinov3_convnext_tiny(
                in_channels=5, num_classes=2
            ),
        }
        missing = [
            environment_name
            for _, environment_name, key in MODEL_SPECS
            if not self.checkpoint_paths.get(key)
        ]
        if missing:
            raise RuntimeError(
                "Checkpoint paths are not configured: " + ", ".join(missing)
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for display_name, _, key in MODEL_SPECS:
            checkpoint_path = Path(self.checkpoint_paths[key]).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model = builders[key]()
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            model.to(device).eval()
            models[display_name] = model

        self._device = device
        self._models = models

    def predict(self, image: np.ndarray) -> list[dict[str, str]]:
        import torch
        import torch.nn.functional as functional

        with self._lock:
            if self._models is None:
                self._load_models()

            height, width = image.shape[-2:]
            padded_height = ((height + 31) // 32) * 32
            padded_width = ((width + 31) // 32) * 32
            tensor = torch.from_numpy(image).unsqueeze(0)
            tensor = functional.pad(
                tensor, (0, padded_width - width, 0, padded_height - height)
            ).to(self._device)

            results = []
            with torch.inference_mode():
                for display_name, model in self._models.items():
                    prediction = model(tensor).argmax(dim=1)[0, :height, :width]
                    results.append(
                        {
                            "name": display_name,
                            "image": mask_data_url(prediction.cpu().numpy()),
                        }
                    )
            return results


def create_app(checkpoint_paths: dict[str, str] | None = None):
    """Create the Flask application without loading models at startup."""
    from flask import Flask, render_template, request

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024
    service = SegmentationService(checkpoint_paths=checkpoint_paths)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/segment")
    def segment():
        upload = request.files.get("satellite_image")
        if upload is None or not upload.filename:
            return render_template("index.html", error="Choose an image to upload."), 400
        try:
            image = read_uploaded_satellite(upload)
            results = service.predict(image)
        except Exception as error:
            return render_template("index.html", error=str(error)), 400
        return render_template(
            "index.html",
            filename=Path(upload.filename).name,
            preview=rgb_preview_data_url(image),
            results=results,
        )

    return app

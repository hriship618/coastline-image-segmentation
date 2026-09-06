# CoastLearn

CoastLearn is a small deep-learning project for extracting shorelines from
satellite imagery and comparing how well different pretrained vision backbones
generalize to unseen coastal regions.

South Florida motivates the project, but the training and evaluation pipeline
is not restricted to Florida.

## Scientific boundary

The learned model will segment currently visible land and water. It will not
directly predict future sea levels. A later visualization may overlay official
sea-level-rise scenario data, clearly labeled as scenario data rather than a
neural-network forecast.

## Build checkpoints

1. Represent a satellite tile and pixel mask with synthetic data.
2. Load real georeferenced image/mask pairs.
3. Train a ResNet34-UNet baseline.
4. Train ConvNeXt-Tiny with a small segmentation head.
5. Train DINOv3 ConvNeXt-Tiny with the same head.
6. Evaluate on geographically held-out coastlines.
7. Convert predicted masks into mapped shoreline vectors.
8. Produce a simple observed-change and scenario visualization.

Each checkpoint will remain independently runnable and documented before the
next one is added.

## Current checkpoint: geographic training and validation

Before adding PyTorch or downloading a large dataset, the code defines exactly
what one training example will contain:

- `image`: floating-point array shaped `[channels, height, width]`
- `mask`: integer array shaped `[height, width]`
- mask value `0`: land
- mask value `1`: water

Run the inspection script from this directory:

```powershell
python scripts/inspect_synthetic.py
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

The synthetic coastline is not training data. It is a fast way to verify the
data shapes and class meanings that every later model will depend on.

The real-data code in `src/coastlearn/data.py` discovers matching SWED GeoTIFF
pairs and loads either RGB or five selected Sentinel-2 bands. No dataset is
stored in this repository. The Colab notebook contains an explicit opt-in
download cell so cloning or importing the project never downloads data by
itself.

`notebooks/02_resnet34_unet.ipynb` constructs the baseline model and performs
one complete optimizer update on synthetic data. This demonstrates the forward
pass, pixel-wise logits, cross-entropy loss, backpropagation, and AdamW update
without downloading satellite data or pretrained weights.

`notebooks/03_convnext_tiny.ipynb` converts a ConvNeXt-Tiny classifier backbone
into a segmentation network. It prints the hierarchical feature-map shapes,
shows the custom convolutional head before and after interpolation, and performs
one optimizer update using separate backbone and head learning rates.

`notebooks/04_dinov3_convnext_tiny.ipynb` loads the gated DINOv3 ConvNeXt-Tiny
weights in Colab, replaces the three-channel RGB stem with a five-channel stem,
verifies how the extra NIR and SWIR weights were initialized, and performs one
segmentation update. Model weights are never downloaded by importing the local
package; loading occurs only when the notebook constructs this model.

`notebooks/05_train_validate.ipynb` groups examples by Sentinel tile so nearby
coastlines cannot leak across splits. It builds DataLoaders, trains one model for
complete epochs, evaluates land and water IoU, saves the best validation
checkpoint, and stops if validation performance no longer improves.

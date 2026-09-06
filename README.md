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

## Project question

Can a water/land segmentation model learned from Sentinel-2 imagery transfer to
coastal regions it did not see during training? The focus is shoreline
extraction, not a claimed prediction of future sea level.

## Data and evaluation protocol

Training uses all 4,334 samples from the
[Sentinel2-NOAA Water Edges Dataset (SNOWED)](https://zenodo.org/records/8112715).
Each sample is a 256 × 256 Sentinel-2 Level-2A array with a binary land/water
label. The models receive five channels: red, green, blue, near infrared (NIR),
and shortwave infrared (SWIR).

SNOWED samples are grouped by the MGRS tile in their original Sentinel product
identifier. Whole geographic tiles—not individual images—are assigned to train,
validation, or internal test. The 98 carefully labeled images in SWED's official
test directory remain outside training and provide a final cross-dataset test.

An early pipeline check split those 98 SWED images into 58/20/20 and produced
promising masks. Those figures are intentionally not reported as final results:
the exercise reused an official benchmark test set and the two models did not
yet share a decoder. Notebook 06 replaces that pilot with the clean protocol.

## Matched model comparison

All three encoders feed the same U-Net-style decoder. It enlarges the deepest
feature map, joins it with three earlier feature maps through skip connections,
and produces two logits at every input pixel.

| Encoder | Initial weights | Decoder |
| --- | --- | --- |
| ResNet-34 | Supervised ImageNet | Shared U-Net decoder |
| ConvNeXt-Tiny | Supervised ImageNet | Shared U-Net decoder |
| ConvNeXt-Tiny | Self-supervised DINOv3 | Shared U-Net decoder |

Ordinary ConvNeXt and DINOv3 ConvNeXt have identical encoder and decoder
structures, isolating the effect of their pretraining. ResNet uses the same
decoder operations and widths, although its encoder feature-channel counts
necessarily differ. Final metrics will be added only after the larger experiment
has run.

## Build steps

1. Verify the image/mask contract with synthetic data.
2. Load real georeferenced image/mask pairs.
3. Train a ResNet-34 U-Net baseline on real data.
4. Train ordinary and DINOv3 ConvNeXt-Tiny with the shared decoder.
5. Evaluate on geographically held-out coastlines.
6. Convert predicted masks into mapped shoreline vectors.
7. Produce a simple observed-change and scenario visualization.

Each notebook is independently runnable in Colab. Start with the data setup
notebook, then run the model-specific notebooks.

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

The real-data code in `src/coastlearn/data.py` loads SWED GeoTIFF pairs and the
NumPy layout used by the official SNOWED archive. It obtains SNOWED geographic
tile IDs by scanning metadata bytes rather than deserializing pickle files. No
dataset is stored in this repository, and downloads are always explicit.

`notebooks/02_resnet34_unet.ipynb` constructs the baseline model and performs
one complete optimizer update on synthetic data. This demonstrates the forward
pass, pixel-wise logits, cross-entropy loss, backpropagation, and AdamW update
without downloading satellite data or pretrained weights.

`notebooks/03_convnext_tiny.ipynb` converts a ConvNeXt-Tiny classifier backbone
into a segmentation network. It prints the hierarchical feature-map shapes,
shows how the shared decoder combines them, and performs one optimizer update
using separate backbone and decoder learning rates.

`notebooks/04_dinov3_convnext_tiny.ipynb` loads the gated DINOv3 ConvNeXt-Tiny
weights in Colab, replaces the three-channel RGB stem with a five-channel stem,
verifies how the extra NIR and SWIR weights were initialized, and performs one
segmentation update. Model weights are never downloaded by importing the local
package; loading occurs only when the notebook constructs this model.

`notebooks/05_train_validate.ipynb` preserves the small SWED pipeline exercise.
It should not be used to produce final benchmark figures.

`notebooks/06_snowed_matched_models.ipynb` is the main experiment. It downloads
SNOWED, extracts only the necessary files, creates a geographic split, trains
one of the three matched models, saves its best checkpoint, and evaluates on
both held-out SNOWED tiles and the external SWED test set.

## Reproducing the experiment

1. Open `notebooks/06_snowed_matched_models.ipynb` in a GPU Colab runtime.
2. Opt in to the 6.8 GB SNOWED download and verify that 4,334 pairs are found.
3. Run the training section separately for `resnet34_unet`, `convnext_tiny`, and
   `dinov3_convnext_tiny`, retaining the same seed and geographic split.
4. Add the internal and external test metrics only after all model choices are
   fixed.

DINOv3 weights require access to the gated Hugging Face model and an `HF_TOKEN`
Colab secret. Datasets and model checkpoints are not committed to this
repository.

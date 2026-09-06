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

## Data and split

The experiments use 98 labeled SWED Sentinel-2 GeoTIFF image/mask pairs. Each
model receives five channels: red, green, blue, near infrared (NIR), and
shortwave infrared (SWIR). The labels are binary: land `0` and water `1`.

Examples are grouped by their Sentinel MGRS tile before splitting, so one tile
can belong to only train, validation, or test. This avoids measuring the model
on near-duplicate neighboring coastlines.

| Split | Images | Sentinel tiles | Purpose |
| --- | ---: | ---: | --- |
| Train | 58 | 23 | Fit model parameters |
| Validation | 20 | 7 | Choose the best checkpoint |
| Test | 20 | 7 | Final, untouched evaluation |

## Results

All figures below are intersection-over-union (IoU), where higher is better.
The test split was not used to select either checkpoint.

| Model configuration | Test mean IoU | Test land IoU | Test water IoU |
| --- | ---: | ---: | ---: |
| ResNet-34 U-Net, five bands | **0.812** | 0.758 | **0.865** |
| DINOv3-pretrained ConvNeXt-Tiny with minimal head, five bands | 0.695 | 0.627 | 0.762 |

The ResNet-34 U-Net is the stronger configuration in this initial experiment.
This does not establish that DINOv3 pretraining is worse: the ResNet setup uses
a full U-Net decoder with skip connections, whereas the DINO experiment uses a
deliberately small decoder. The comparison therefore measures complete model
configurations, not pretraining alone.

## Build steps

1. Verify the image/mask contract with synthetic data.
2. Load real georeferenced image/mask pairs.
3. Train a ResNet-34 U-Net baseline on real data.
4. Train DINOv3 ConvNeXt-Tiny with a small custom segmentation head.
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

## Reproducing the experiment

1. Run `notebooks/01_colab_data_setup.ipynb` in Colab and opt in to the full
   SWED download.
2. Run the ResNet training section with `epochs=10`.
3. Evaluate the saved checkpoint on the untouched test split.
4. For DINOv3, use a GPU runtime, provide a Hugging Face token with access to
   the gated weights, and repeat the same fixed geographic split.

The full dataset download contains supporting files beyond the 98 labeled
GeoTIFF pairs used by this experiment. No dataset or model checkpoint is stored
in this repository.

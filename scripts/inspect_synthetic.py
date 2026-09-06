"""Print the contents of one synthetic CoastLearn example."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.synthetic import LAND_CLASS, WATER_CLASS, make_synthetic_coast


def main() -> None:
    image, mask = make_synthetic_coast()
    water_fraction = float((mask == WATER_CLASS).mean())

    print(f"image shape: {image.shape}  # channels, height, width")
    print(f"image dtype: {image.dtype}")
    print(f"image range: {image.min():.3f} to {image.max():.3f}")
    print(f"mask shape:  {mask.shape}  # height, width")
    print(f"mask dtype:  {mask.dtype}")
    print(f"classes:     land={LAND_CLASS}, water={WATER_CLASS}")
    print(f"water area:  {water_fraction:.1%} of pixels")


if __name__ == "__main__":
    main()


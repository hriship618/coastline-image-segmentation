"""Dependency-light tests for SWED file discovery and configuration."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coastlearn.data import (
    FIVE_BANDS,
    RGB_BANDS,
    ImageMaskPair,
    discover_snowed_pairs,
    discover_swed_pairs,
    read_snowed_image,
    read_snowed_mask,
    snowed_region_id,
    split_pairs_by_region,
    swed_region_id,
    validate_band_positions,
)


class SwedDiscoveryTests(unittest.TestCase):
    def test_discovers_matching_pair_in_separate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            image = images / "scene_image_0_0.tif"
            label = labels / "scene_label_0_0.tif"
            image.touch()
            label.touch()

            pairs = discover_swed_pairs(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].image_path, image)
            self.assertEqual(pairs[0].mask_path, label)

    def test_missing_label_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "scene_image_0_0.tif").touch()

            with self.assertRaisesRegex(ValueError, "Could not find exactly one label"):
                discover_swed_pairs(root)

    def test_band_presets_are_valid_for_twelve_band_image(self) -> None:
        self.assertEqual(validate_band_positions(RGB_BANDS, 12), RGB_BANDS)
        self.assertEqual(validate_band_positions(FIVE_BANDS, 12), FIVE_BANDS)

    def test_out_of_range_band_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed raster"):
            validate_band_positions((2, 13), available_bands=12)

    def test_region_is_read_from_sentinel_tile(self) -> None:
        pair = ImageMaskPair(
            Path("S2A_20190312_T48QYJ_scene_image_0_0.tif"),
            Path("S2A_20190312_T48QYJ_scene_label_0_0.tif"),
        )
        self.assertEqual(swed_region_id(pair), "48QYJ")

    def test_geographic_splits_have_no_region_overlap(self) -> None:
        pairs = []
        for tile in ("10AAA", "11BBB", "12CCC", "13DDD", "14EEE"):
            for index in range(2):
                pairs.append(
                    ImageMaskPair(
                        Path(f"S2A_T{tile}_scene_image_{index}_0.tif"),
                        Path(f"S2A_T{tile}_scene_label_{index}_0.tif"),
                    )
                )

        splits = split_pairs_by_region(pairs, seed=19)
        region_sets = [
            {swed_region_id(pair) for pair in split}
            for split in (splits.train, splits.validation, splits.test)
        ]
        self.assertTrue(region_sets[0].isdisjoint(region_sets[1]))
        self.assertTrue(region_sets[0].isdisjoint(region_sets[2]))
        self.assertTrue(region_sets[1].isdisjoint(region_sets[2]))
        self.assertEqual(sum(len(split) for split in (splits.train, splits.validation, splits.test)), 10)

    def test_discovers_snowed_numpy_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sample = Path(temporary_directory) / "SNOWED" / "42"
            sample.mkdir(parents=True)
            (sample / "sample_2A.npy").touch()
            (sample / "label.npy").touch()
            (sample / "metadata.pkl").touch()

            pairs = discover_snowed_pairs(temporary_directory)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].image_path, sample / "sample_2A.npy")

    def test_reads_snowed_region_without_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sample = Path(temporary_directory)
            image = sample / "sample_2A.npy"
            image.touch()
            (sample / "metadata.pkl").write_bytes(
                b"untrusted pickle bytes S2B_SCENE_T06VUM_20200911"
            )

            pair = ImageMaskPair(image, sample / "label.npy")
            self.assertEqual(snowed_region_id(pair), "06VUM")

    def test_reads_and_orients_snowed_arrays(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "sample_2A.npy"
            mask_path = root / "label.npy"
            image = np.zeros((2, 3, 12), dtype=np.uint16)
            for channel in range(12):
                image[..., channel] = (channel + 1) * 100
            np.save(image_path, image)
            np.save(mask_path, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8))

            loaded_image = read_snowed_image(image_path)
            loaded_mask = read_snowed_mask(mask_path)

            self.assertEqual(loaded_image.shape, (5, 2, 3))
            np.testing.assert_allclose(
                loaded_image[:, 0, 0], np.array([0.04, 0.03, 0.02, 0.08, 0.11])
            )
            np.testing.assert_array_equal(
                loaded_mask, np.array([[1, 1, 1], [0, 0, 0]])
            )


if __name__ == "__main__":
    unittest.main()

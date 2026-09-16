import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from segmentation_benchmark.cli import main
from segmentation_benchmark.io import as_instance_labels
from segmentation_benchmark.metrics import compare_instances, object_measurements
from segmentation_benchmark.methods import OtsuMethod, OtsuWatershedMethod


class InstanceMetricsTests(unittest.TestCase):
    def test_binary_mask_is_converted_to_instances(self):
        binary = np.zeros((6, 6, 6), dtype=np.uint8)
        binary[1:3, 1:3, 1:3] = 255
        binary[4:6, 4:6, 4:6] = 255

        labels = as_instance_labels(binary, mode="auto")

        self.assertEqual(int(labels.max()), 2)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})

    def test_object_metrics_report_missed_ground_truth(self):
        ground_truth = np.zeros((8, 8, 8), dtype=np.uint16)
        ground_truth[1:3, 1:3, 1:3] = 1
        ground_truth[5:7, 5:7, 5:7] = 2
        predicted = np.zeros_like(ground_truth)
        predicted[1:3, 1:3, 1:3] = 1

        metrics, accepted, best = compare_instances(
            predicted, ground_truth, iou_threshold=0.5
        )
        objects = object_measurements(
            predicted,
            image_name="sample.tif",
            method="example",
            voxel_size=(2.0, 1.0, 1.0),
            accepted_matches=accepted,
            best_matches=best,
        )

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["object_recall"], 0.5)
        self.assertEqual(objects[0]["voxel_count"], 8)
        self.assertEqual(objects[0]["volume_um3"], 16.0)
        self.assertEqual(objects[0]["matched_ground_truth_label"], 1)

    def test_watershed_splits_a_touching_otsu_component(self):
        z, y, x = np.ogrid[:24, :24, :24]
        first = (z - 12) ** 2 + (y - 9) ** 2 + (x - 12) ** 2 <= 25
        second = (z - 12) ** 2 + (y - 15) ** 2 + (x - 12) ** 2 <= 25
        volume = np.zeros((24, 24, 24), dtype=np.float32)
        volume[first | second] = 1.0

        connected = OtsuMethod().predict(volume, Path("sample.tif"))
        watershed = OtsuWatershedMethod(min_distance=3).predict(
            volume, Path("sample.tif")
        )

        self.assertEqual(int(connected.max()), 1)
        self.assertEqual(int(watershed.max()), 2)


class BenchmarkCliTests(unittest.TestCase):
    def test_otsu_run_writes_masks_and_metric_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            ground_truth_dir = root / "ground_truth"
            output_dir = root / "run"
            input_dir.mkdir()
            ground_truth_dir.mkdir()

            volume = np.zeros((12, 12, 12), dtype=np.uint16)
            volume[1:4, 1:4, 1:4] = 100
            volume[7:10, 7:10, 7:10] = 200
            ground_truth = (volume > 0).astype(np.uint8) * 255
            tifffile.imwrite(input_dir / "sample.tif", volume)
            tifffile.imwrite(ground_truth_dir / "sample.tif", ground_truth)

            main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--ground-truth-dir",
                    str(ground_truth_dir),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "otsu",
                    "--voxel-size",
                    "2",
                    "1",
                    "1",
                ]
            )

            predicted = tifffile.imread(
                output_dir / "masks" / "otsu" / "sample.tif"
            )
            self.assertEqual(int(predicted.max()), 2)
            self.assertTrue(
                (output_dir / "masks" / "ground_truth" / "sample.tif").is_file()
            )
            self.assertTrue((output_dir / "summary.md").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())

            with (output_dir / "metrics" / "by_volume.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["num_predicted_masks"], "2")
            self.assertEqual(row["num_ground_truth_masks"], "2")
            self.assertEqual(float(row["dice"]), 1.0)

            with (output_dir / "metrics" / "by_object.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)

            precomputed_dir = root / "precomputed"
            precomputed_dir.mkdir()
            tifffile.imwrite(precomputed_dir / "sample.tif", ground_truth)
            precomputed_output = root / "precomputed-run"
            main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--ground-truth-dir",
                    str(ground_truth_dir),
                    "--output-dir",
                    str(precomputed_output),
                    "--methods",
                    "--precomputed",
                    f"existing={precomputed_dir}",
                ]
            )
            with (precomputed_output / "metrics" / "by_volume.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                precomputed_row = next(csv.DictReader(handle))
            self.assertEqual(precomputed_row["method"], "existing")
            self.assertEqual(precomputed_row["runtime_seconds"], "")


if __name__ == "__main__":
    unittest.main()

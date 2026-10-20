"""Score a directory of predicted instance masks against matching ground
truth (binary Ilastik masks, converted to instance labels here), at both
IoU thresholds. Writes a per-volume CSV and prints a region-level summary.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.io_utils import load_ground_truth_as_labels, load_volume  # noqa: E402
from common.metrics import compare_instances  # noqa: E402

IOU_THRESHOLDS = (0.01, 0.5)


def flatten_row(chunk, metrics, runtime_by_chunk):
    row = {
        "chunk": chunk,
        "runtime_seconds": runtime_by_chunk.get(chunk),
        "num_predicted_masks": metrics["num_predicted_masks"],
        "num_ground_truth_masks": metrics["num_ground_truth_masks"],
        "count_error": metrics["count_error"],
        "dice": metrics["dice"],
        "semantic_iou": metrics["semantic_iou"],
    }
    for threshold in IOU_THRESHOLDS:
        stats = metrics[f"iou_{threshold}"]
        suffix = str(threshold).replace(".", "")
        for key, value in stats.items():
            row[f"{key}_iou{suffix}"] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--runtime-csv",
        help="Optional run_inference.py runtime CSV to merge in by chunk id",
    )
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    gt_dir = Path(args.ground_truth_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    runtime_by_chunk = {}
    if args.runtime_csv:
        with open(args.runtime_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                runtime_by_chunk[row["chunk"]] = float(row["runtime_seconds"])

    mask_files = sorted(masks_dir.glob("*.tif"))
    print(f"Found {len(mask_files)} predicted masks in {masks_dir}")

    rows = []
    for index, mask_path in enumerate(mask_files, start=1):
        gt_path = gt_dir / mask_path.name
        if not gt_path.exists():
            print(f"  [{index}/{len(mask_files)}] {mask_path.name}: no matching ground truth, skipping")
            continue
        predicted = load_volume(mask_path)
        ground_truth = load_ground_truth_as_labels(gt_path)
        metrics = compare_instances(predicted, ground_truth, iou_thresholds=IOU_THRESHOLDS)
        rows.append(flatten_row(mask_path.stem, metrics, runtime_by_chunk))
        print(f"  [{index}/{len(mask_files)}] {mask_path.name}: "
              f"pred={metrics['num_predicted_masks']} gt={metrics['num_ground_truth_masks']} "
              f"f1@0.5={metrics['iou_0.5']['f1']:.3f}")

    if not rows:
        print("No matched pairs found; nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")

    print("\nRegion summary:")
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "")
        f1_values = [row[f"f1_iou{suffix}"] for row in rows]
        precision_values = [row[f"precision_iou{suffix}"] for row in rows]
        recall_values = [row[f"recall_iou{suffix}"] for row in rows]
        print(
            f"  IoU={threshold}: mean F1={statistics.fmean(f1_values):.4f} "
            f"mean precision={statistics.fmean(precision_values):.4f} "
            f"mean recall={statistics.fmean(recall_values):.4f}"
        )
    dice_values = [row["dice"] for row in rows]
    print(f"  mean Dice={statistics.fmean(dice_values):.4f}")
    runtimes = [row["runtime_seconds"] for row in rows if row["runtime_seconds"] is not None]
    if runtimes:
        print(f"  mean runtime={statistics.fmean(runtimes):.2f}s over {len(runtimes)} volumes")


if __name__ == "__main__":
    main()

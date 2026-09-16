import argparse
import csv
import gc
import json
import platform
import re
import statistics
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

from . import __version__
from .io import (
    as_instance_labels,
    discover_volumes,
    load_mask,
    load_volume,
    matching_mask_path,
    write_label_mask,
)
from .methods import (
    CellposeMethod,
    OtsuMethod,
    OtsuWatershedMethod,
    PrecomputedMethod,
    StarDistMethod,
    UNetMethod,
)
from .metrics import compare_instances, object_measurements


VOLUME_FIELDS = [
    "image",
    "method",
    "input_path",
    "ground_truth_path",
    "mask_path",
    "runtime_seconds",
    "shape_z",
    "shape_y",
    "shape_x",
    "num_predicted_masks",
    "num_ground_truth_masks",
    "count_error",
    "absolute_count_error",
    "predicted_foreground_voxels",
    "ground_truth_foreground_voxels",
    "foreground_fraction",
    "dice",
    "semantic_iou",
    "true_positives",
    "false_positives",
    "false_negatives",
    "object_precision",
    "object_recall",
    "object_f1",
    "mean_matched_iou",
    "panoptic_quality",
]

OBJECT_FIELDS = [
    "image",
    "method",
    "mask_label",
    "voxel_count",
    "volume_um3",
    "equivalent_sphere_diameter_um",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "bbox_z_min",
    "bbox_y_min",
    "bbox_x_min",
    "bbox_z_max",
    "bbox_y_max",
    "bbox_x_max",
    "matched_ground_truth_label",
    "matched_iou",
    "best_ground_truth_label",
    "best_iou",
]

SUMMARY_FIELDS = [
    "method",
    "num_volumes",
    "setup_seconds",
    "inference_total_seconds",
    "inference_mean_seconds",
    "setup_plus_inference_seconds",
    "total_masks",
    "mean_masks_per_volume",
    "mean_mask_size_voxels",
    "median_mask_size_voxels",
    "mean_dice",
    "mean_semantic_iou",
    "mean_object_precision",
    "mean_object_recall",
    "mean_object_f1",
    "mean_panoptic_quality",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run 3-D nuclei segmentation methods on the same TIFF test volumes, "
            "store instance masks, and compare them with matching ground truth."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory of test TIFF volumes")
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        help="Directory of binary or instance-labeled ground-truth TIFF masks",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New, empty directory for masks, metrics, and the run manifest",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=["otsu", "otsu-watershed", "cellpose", "stardist", "unet"],
        default=["otsu", "otsu-watershed"],
        help="Methods to execute (default: otsu otsu-watershed)",
    )
    parser.add_argument(
        "--precomputed",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help=(
            "Include masks generated elsewhere. May be repeated. Runtime is left "
            "blank because segmentation is not executed."
        ),
    )
    parser.add_argument(
        "--ground-truth-mode",
        choices=["auto", "binary", "labels"],
        default="auto",
        help="Interpret ground-truth TIFFs as binary or instance labels (default: auto)",
    )
    parser.add_argument(
        "--precomputed-mask-mode",
        choices=["auto", "binary", "labels"],
        default="auto",
        help="Interpret precomputed masks as binary or instance labels (default: auto)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("Z_UM", "Y_UM", "X_UM"),
        help="Voxel size in micrometers; enables physical object sizes",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU for an object-level true positive (default: 0.5)",
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel in 4-D inputs")
    parser.add_argument(
        "--channel-axis",
        type=int,
        default=0,
        help="Channel axis in 4-D inputs (default: 0)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Remove smaller Otsu objects, in voxels (default: disabled)",
    )
    parser.add_argument(
        "--watershed-min-distance",
        type=int,
        default=3,
        help="Minimum seed separation for Otsu watershed, in voxels (default: 3)",
    )
    parser.add_argument(
        "--dark-foreground",
        action="store_true",
        help="Treat values below the Otsu threshold as nuclei",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for learned methods (default: auto)",
    )
    parser.add_argument("--unet-model", help="Path to a trained U-Net .pth or .ckpt")
    parser.add_argument("--unet-layers", type=int, choices=[3, 4], default=3)
    parser.add_argument("--unet-chunk-size", type=int, default=850)
    parser.add_argument("--unet-patch-size", type=int, default=128)
    parser.add_argument("--unet-threshold", type=float, default=0.5)
    parser.add_argument("--cellpose-model", default="nuclei")
    parser.add_argument("--cellpose-model-path")
    parser.add_argument("--cellpose-diameter", type=float)
    parser.add_argument("--stardist-model", default="3D_demo")
    parser.add_argument("--stardist-model-path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _validate_args(args)
    input_root = Path(args.input_dir).resolve()
    ground_truth_root = Path(args.ground_truth_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    input_paths = discover_volumes(input_root)
    relative_paths = [path.relative_to(input_root) for path in input_paths]
    ground_truth_paths = {
        relative: matching_mask_path(ground_truth_root, relative)
        for relative in relative_paths
    }
    precomputed = _parse_precomputed(args.precomputed)
    method_names = list(args.methods) + list(precomputed)
    if not method_names:
        raise ValueError("Select at least one --methods entry or --precomputed mask set")
    _validate_method_names(method_names)
    _prepare_output(output_root)

    volume_rows = []
    object_rows = []
    setup_times = {}
    print(f"Found {len(input_paths)} test volume(s)")

    for relative in relative_paths:
        labels = load_mask(
            ground_truth_paths[relative], mode=args.ground_truth_mode
        )
        output_path = output_root / "masks" / "ground_truth" / relative
        write_label_mask(output_path, labels)
        object_rows.extend(
            object_measurements(
                labels,
                image_name=relative.as_posix(),
                method="ground_truth",
                voxel_size=args.voxel_size,
            )
        )

    for method_name in method_names:
        print(f"Running {method_name}")
        setup_start = perf_counter()
        try:
            method = _build_method(method_name, precomputed, args)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize method {method_name}: {exc}"
            ) from exc
        setup_times[method_name] = perf_counter() - setup_start

        for index, (input_path, relative) in enumerate(
            zip(input_paths, relative_paths), start=1
        ):
            print(f"  [{index}/{len(input_paths)}] {relative.as_posix()}")
            volume = load_volume(
                input_path,
                channel=args.channel,
                channel_axis=args.channel_axis,
            )
            ground_truth = load_mask(
                ground_truth_paths[relative], mode=args.ground_truth_mode
            )
            if volume.shape != ground_truth.shape:
                raise ValueError(
                    f"Input {relative} has shape {volume.shape}, but its ground truth "
                    f"has shape {ground_truth.shape}"
                )

            try:
                _synchronize(method)
                started = perf_counter()
                predicted = method.predict(volume, relative)
                _synchronize(method)
            except Exception as exc:
                raise RuntimeError(
                    f"Method {method_name} failed on {relative.as_posix()}: {exc}"
                ) from exc
            runtime = perf_counter() - started if method.reports_runtime else None
            predicted = as_instance_labels(predicted, mode="labels")
            if predicted.shape != volume.shape:
                raise ValueError(
                    f"Method {method_name} returned shape {predicted.shape} for "
                    f"{relative}, expected {volume.shape}"
                )

            mask_path = output_root / "masks" / method_name / relative
            write_label_mask(mask_path, predicted)
            metrics, accepted_matches, best_matches = compare_instances(
                predicted,
                ground_truth,
                iou_threshold=args.iou_threshold,
            )
            volume_rows.append(
                {
                    "image": relative.as_posix(),
                    "method": method_name,
                    "input_path": str(input_path),
                    "ground_truth_path": str(ground_truth_paths[relative]),
                    "mask_path": str(mask_path),
                    "runtime_seconds": runtime,
                    "shape_z": volume.shape[0],
                    "shape_y": volume.shape[1],
                    "shape_x": volume.shape[2],
                    **metrics,
                }
            )
            object_rows.extend(
                object_measurements(
                    predicted,
                    image_name=relative.as_posix(),
                    method=method_name,
                    voxel_size=args.voxel_size,
                    accepted_matches=accepted_matches,
                    best_matches=best_matches,
                )
            )

        del method
        gc.collect()

    summary_rows = _summarize(volume_rows, object_rows, setup_times, method_names)
    metrics_root = output_root / "metrics"
    _write_csv(metrics_root / "by_volume.csv", VOLUME_FIELDS, volume_rows)
    _write_csv(metrics_root / "by_object.csv", OBJECT_FIELDS, object_rows)
    _write_csv(metrics_root / "by_method.csv", SUMMARY_FIELDS, summary_rows)
    _write_summary(output_root / "summary.md", summary_rows, args.iou_threshold)
    _write_manifest(
        output_root / "manifest.json",
        args,
        input_paths,
        setup_times,
        method_names,
    )
    print(f"Benchmark complete: {output_root}")
    return output_root


def _validate_args(args):
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1")
    if args.min_size < 0:
        raise ValueError("--min-size cannot be negative")
    if args.watershed_min_distance < 1:
        raise ValueError("--watershed-min-distance must be at least 1")
    if args.unet_chunk_size < 1 or args.unet_patch_size < 1:
        raise ValueError("U-Net chunk and patch sizes must be positive")
    if args.unet_patch_size % (2 ** args.unet_layers) != 0:
        raise ValueError(
            "--unet-patch-size must be divisible by 2 to the power of "
            "--unet-layers"
        )
    if args.voxel_size is not None and any(value <= 0 for value in args.voxel_size):
        raise ValueError("--voxel-size values must be positive")
    if "unet" in args.methods and not args.unet_model:
        raise ValueError("--unet-model is required when selecting the unet method")


def _parse_precomputed(specifications):
    parsed = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid --precomputed value {specification!r}; use NAME=DIR"
            )
        name, directory = specification.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError(
                f"Invalid precomputed method name {name!r}; use lowercase letters, "
                "digits, underscores, or hyphens"
            )
        if name in parsed:
            raise ValueError(f"Duplicate precomputed method name: {name}")
        parsed[name] = Path(directory).resolve()
    return parsed


def _validate_method_names(method_names):
    duplicates = sorted(
        name for name in set(method_names) if method_names.count(name) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate method names: {', '.join(duplicates)}")
    if "ground_truth" in method_names:
        raise ValueError("ground_truth is reserved and cannot be a method name")


def _prepare_output(output_root):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory must be new or empty to avoid mixing runs: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _build_method(name, precomputed, args):
    if name == "otsu":
        return OtsuMethod(
            min_size=args.min_size,
            dark_foreground=args.dark_foreground,
        )
    if name == "otsu-watershed":
        return OtsuWatershedMethod(
            min_size=args.min_size,
            min_distance=args.watershed_min_distance,
            dark_foreground=args.dark_foreground,
        )
    if name == "cellpose":
        return CellposeMethod(
            model_name=args.cellpose_model,
            model_path=args.cellpose_model_path,
            diameter=args.cellpose_diameter,
            device=args.device,
        )
    if name == "stardist":
        return StarDistMethod(
            model_name=args.stardist_model,
            model_path=args.stardist_model_path,
        )
    if name == "unet":
        return UNetMethod(
            model_path=args.unet_model,
            num_layers=args.unet_layers,
            chunk_size=args.unet_chunk_size,
            patch_size=args.unet_patch_size,
            threshold=args.unet_threshold,
            device=args.device,
        )
    return PrecomputedMethod(
        name,
        precomputed[name],
        mask_mode=args.precomputed_mask_mode,
    )


def _summarize(volume_rows, object_rows, setup_times, method_names):
    summaries = []
    for method_name in method_names:
        method_volumes = [
            row for row in volume_rows if row["method"] == method_name
        ]
        method_objects = [
            row for row in object_rows if row["method"] == method_name
        ]
        runtimes = [
            row["runtime_seconds"]
            for row in method_volumes
            if row["runtime_seconds"] is not None
        ]
        sizes = [row["voxel_count"] for row in method_objects]
        summaries.append(
            {
                "method": method_name,
                "num_volumes": len(method_volumes),
                "setup_seconds": setup_times[method_name],
                "inference_total_seconds": sum(runtimes) if runtimes else None,
                "inference_mean_seconds": _mean(runtimes),
                "setup_plus_inference_seconds": (
                    setup_times[method_name] + sum(runtimes)
                    if runtimes
                    else None
                ),
                "total_masks": sum(
                    row["num_predicted_masks"] for row in method_volumes
                ),
                "mean_masks_per_volume": _mean(
                    [row["num_predicted_masks"] for row in method_volumes]
                ),
                "mean_mask_size_voxels": _mean(sizes),
                "median_mask_size_voxels": (
                    statistics.median(sizes) if sizes else None
                ),
                "mean_dice": _mean([row["dice"] for row in method_volumes]),
                "mean_semantic_iou": _mean(
                    [row["semantic_iou"] for row in method_volumes]
                ),
                "mean_object_precision": _mean(
                    [row["object_precision"] for row in method_volumes]
                ),
                "mean_object_recall": _mean(
                    [row["object_recall"] for row in method_volumes]
                ),
                "mean_object_f1": _mean(
                    [row["object_f1"] for row in method_volumes]
                ),
                "mean_panoptic_quality": _mean(
                    [row["panoptic_quality"] for row in method_volumes]
                ),
            }
        )
    return summaries


def _mean(values):
    return statistics.fmean(values) if values else None


def _synchronize(method):
    synchronize = getattr(method, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, rows, iou_threshold):
    lines = [
        "# Segmentation benchmark",
        "",
        (
            "Object precision, recall, and F1 use one-to-one greedy matching at "
            f"IoU >= {iou_threshold:g}. Runtime excludes TIFF I/O, metric "
            "calculation, and mask writing."
        ),
        "",
        "| Method | Volumes | Masks | Inference (s) | Dice | Object F1 | Panoptic quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {num_volumes} | {total_masks} | {runtime} | "
            "{dice} | {f1} | {pq} |".format(
                method=row["method"],
                num_volumes=row["num_volumes"],
                total_masks=row["total_masks"],
                runtime=_format_number(row["inference_total_seconds"]),
                dice=_format_number(row["mean_dice"]),
                f1=_format_number(row["mean_object_f1"]),
                pq=_format_number(row["mean_panoptic_quality"]),
            )
        )
    lines.extend(
        [
            "",
            "Detailed metrics are in `metrics/by_volume.csv`, "
            "`metrics/by_object.csv`, and `metrics/by_method.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value):
    return "-" if value is None else f"{value:.4f}"


def _write_manifest(path, args, input_paths, setup_times, method_names):
    manifest = {
        "schema_version": 1,
        "benchmark_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "command": sys.argv,
        "methods": method_names,
        "method_setup_seconds": setup_times,
        "num_test_volumes": len(input_paths),
        "test_volumes": [str(item) for item in input_paths],
        "arguments": vars(args),
        "timing_definition": (
            "Per-volume runtime covers method.predict only. It excludes input and "
            "ground-truth reads, output writes, and metric calculation. Model "
            "construction and weight loading are reported separately."
        ),
        "object_matching": (
            "Predictions and ground truth are greedily matched one-to-one in "
            "descending pairwise IoU order at the configured threshold."
        ),
    }
    path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _package_versions():
    packages = {}
    for distribution in (
        "numpy",
        "scipy",
        "scikit-image",
        "tifffile",
        "torch",
        "cellpose",
        "stardist",
        "tensorflow",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return packages

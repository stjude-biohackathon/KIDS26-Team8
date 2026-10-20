"""Run Cellpose-SAM inference over a directory of raw volumes, writing
instance-labeled masks plus per-volume runtime/peak-memory to a CSV.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from cellpose import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_volume, write_label_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--masks-out-dir", required=True)
    parser.add_argument("--runtime-csv", required=True)
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Leave unset. Cellpose-SAM is scale-invariant by design; measured on "
        "pfCortex chunks, setting any diameter reduced both accuracy and speed.",
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Lower it (e.g. -1, -2, -3) to recover more/smaller masks when the "
        "model under-segments. Default 0.0.",
    )
    parser.add_argument(
        "--tile-norm-blocksize",
        type=int,
        default=0,
        help="0 normalizes the whole volume as one, which lets bright regions "
        "crush dim ones. 100-200 normalizes in local blocks instead, for data "
        "with inhomogeneous brightness (light-sheet varies with depth).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    masks_out_dir = Path(args.masks_out_dir)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    runtime_csv = Path(args.runtime_csv)
    runtime_csv.parent.mkdir(parents=True, exist_ok=True)

    gpu = args.device == "cuda"
    model = models.CellposeModel(gpu=gpu, pretrained_model=args.model)

    raw_files = sorted(raw_dir.glob("*.tif"))
    print(f"Found {len(raw_files)} volumes in {raw_dir}")

    rows = []
    for index, path in enumerate(raw_files, start=1):
        print(f"[{index}/{len(raw_files)}] {path.name}")
        volume = load_volume(path)

        if gpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        masks, flows, styles = model.eval(
            volume,
            do_3D=True,
            z_axis=0,
            diameter=args.diameter,
            cellprob_threshold=args.cellprob_threshold,
            normalize={"tile_norm_blocksize": args.tile_norm_blocksize},
        )
        if gpu:
            torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6 if gpu else None

        write_label_mask(masks_out_dir / path.name, masks)
        rows.append(
            {
                "chunk": path.stem,
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": peak_memory_mb,
                "shape_z": volume.shape[0],
                "shape_y": volume.shape[1],
                "shape_x": volume.shape[2],
                "num_predicted_masks": int(masks.max()),
                "diameter": args.diameter,
                "cellprob_threshold": args.cellprob_threshold,
                "tile_norm_blocksize": args.tile_norm_blocksize,
            }
        )

    with runtime_csv.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {runtime_csv}")


if __name__ == "__main__":
    main()

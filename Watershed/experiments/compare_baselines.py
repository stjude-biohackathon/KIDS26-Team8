"""
Score several segmentations of ONE ground-truth chunk side by side, using the
same metrics as nuclei_stage1/evaluate.py (Hungarian match at IoU>=0.25).

Methods:
  old_watershed      the ORIGINAL stage 1: isotropic voxels (1,1,1), min_seed_distance=10,
                     no deskew, no anisotropic footprint. (config values from before voxel_size
                     was wired in.)
  current_native     current config.py (voxel_size, min_seed_distance=17), native chunk
  current_deskewed   current config.py on the deskewed chunk (GT deskewed to match)
  cellpose_sam       Cellpose-SAM, 3D, native chunk (only if `cellpose` v4+ is installed; use
                     --skip-cellpose to leave it out). Needs a GPU to be practical.

Each method is scored against GT in its own frame (deskewed GT for the deskewed run), so no
labels are resampled onto a different grid.

Usage:
    python experiments/compare_baselines.py --raw data\\ground_truth\\gt_chunk1_raw.tif ^
        --gt data\\ground_truth\\ground_truth_chunk1.tif --z-range 1:127
    python experiments/compare_baselines.py ... --skip-cellpose
    python experiments/compare_baselines.py ... --save-prefix data\\outputs\\cmp_chunk1   # also writes label tifs
"""
import argparse
from dataclasses import replace

import numpy as np
import tifffile

from nuclei_stage1.evaluate import evaluate_one, parse_z_range
from nuclei_stage1.config import Stage1Config
from nuclei_stage1.deskew import deskew_volume
from nuclei_stage1.segmentation import run_stage1

KEYS = ["pred_count", "gt_count", "matched", "missed_false_negatives", "extra_false_positives",
        "split_gt_nuclei", "merged_pred_objects", "precision", "recall", "f1", "mean_matched_iou"]


def run_cellpose(raw, voxel_size):
    from cellpose import models                      # v4+ = Cellpose-SAM
    model = models.CellposeModel(gpu=True)
    aniso = float(voxel_size[0]) / float(min(voxel_size[1], voxel_size[2]))
    masks, _, _ = model.eval(raw, do_3D=True, z_axis=0, channel_axis=None,
                             anisotropy=aniso, batch_size=8)
    return masks.astype(np.int32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--z-range", default=None)
    ap.add_argument("--skip-cellpose", action="store_true")
    ap.add_argument("--save-prefix", default=None)
    a = ap.parse_args()

    cfg = Stage1Config()
    old_cfg = replace(cfg, voxel_size=(1.0, 1.0, 1.0), min_seed_distance=10)

    raw = tifffile.imread(a.raw)
    gt = tifffile.imread(a.gt).astype(np.int32)
    z0, z1 = parse_z_range(a.z_range, gt.shape[0])
    raw_d, valid = deskew_volume(raw)
    gt_d, _ = deskew_volume(gt, is_labels=True)

    runs = {
        "old_watershed": (run_stage1(raw, old_cfg)[0], gt),
        "current_native": (run_stage1(raw, cfg)[0], gt),
        "current_deskewed": (run_stage1(raw_d, cfg, valid=valid)[0], gt_d),
    }
    if not a.skip_cellpose:
        try:
            runs["cellpose_sam"] = (run_cellpose(raw, cfg.voxel_size), gt)
        except Exception as e:                                       # noqa: BLE001
            print(f"[cellpose_sam skipped: {type(e).__name__}: {e}]")

    res = {}
    for name, (lab, g) in runs.items():
        res[name] = evaluate_one(lab[z0:z1], g[z0:z1], 0.25, 0.15)
        if a.save_prefix:
            tifffile.imwrite(f"{a.save_prefix}_{name}.tif", lab.astype(np.uint32))

    print(f"\nvs ground truth, z=[{z0}:{z1}]   (old = isotropic voxels, seed dist 10, no deskew; "
          f"current = voxel_size {cfg.voxel_size}, seed dist {cfg.min_seed_distance})")
    print(f"{'':>24}" + "".join(f"{n:>18}" for n in res))
    for k in KEYS:
        print(f"{k:>24}" + "".join(
            (f"{res[n][k]:>18.3f}" if isinstance(res[n][k], float) else f"{res[n][k]:>18}") for n in res))

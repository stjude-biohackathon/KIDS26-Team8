"""
Denoise ONLY the seed-finding landscape; the mask (and so the object boundaries) always comes from
the unsmoothed raw. Whole-image pre-filtering lowered IoU because it blurred boundaries
(see sweep_denoise.py); here the labels are always cut from the raw-derived mask.

Variants of the landscape that peak_local_max finds seeds on and the watershed floods:
  baseline          -distance of the raw mask (current stage 1)
  dt_smooth=S       Gaussian-smoothed distance map (sigma S um)
  int_blend=S       distance x smoothed-intensity (sigma S um, scaled 0..1 inside the mask): seeds
                    prefer bright cores, not just mask-thick spots
  dt_from_smooth=S  distance of a mask recomputed from the smoothed image (cleaner shape for seeds);
                    watershed still floods inside the raw mask
All classify/min-size steps are identical to run_stage1.

Usage:
    python experiments/sweep_seed_landscape.py --raw data\\ground_truth\\gt_chunk1_raw.tif ^
        --gt data\\ground_truth\\ground_truth_chunk1.tif --z-range 1:127
"""
import argparse
import time

import numpy as np
import tifffile

from nuclei_stage1 import segmentation as s1
from nuclei_stage1.evaluate import evaluate_one, parse_z_range
from nuclei_stage1.config import Stage1Config

ndi = s1.ndi


def vox(s_um, vs):
    return tuple(s_um / v for v in vs)


def landscape(raw, mask, cfg, kind, s):
    vs = cfg.voxel_size
    dist = ndi.distance_transform_edt(mask, sampling=vs)
    if kind == "baseline":
        return dist
    if kind == "dt_smooth":
        return ndi.gaussian_filter(dist, vox(s, vs))
    sm = ndi.gaussian_filter(raw.astype(np.float32), vox(s, vs))
    if kind == "int_blend":
        lo, hi = np.percentile(sm[mask], [5, 99]) if mask.any() else (0, 1)
        return dist * np.clip((sm - lo) / max(hi - lo, 1e-6), 0, 1)
    if kind == "dt_from_smooth":
        m2 = s1.foreground_mask(sm, mode=cfg.threshold_mode,
                                target_foreground_percentile=cfg.target_foreground_percentile)
        return ndi.distance_transform_edt(m2, sampling=vs)
    raise ValueError(kind)


def segment(raw, cfg, kind, s=0.0):
    mask = s1.foreground_mask(raw, mode=cfg.threshold_mode,
                              target_foreground_percentile=cfg.target_foreground_percentile)
    land = landscape(raw, mask, cfg, kind, s)
    fp = s1._anisotropic_seed_footprint(cfg.min_seed_distance, cfg.voxel_size)
    coords = s1.peak_local_max(land, footprint=fp, labels=mask, exclude_border=False,
                               threshold_rel=cfg.seed_prominence)
    seeds = np.zeros(mask.shape, bool)
    if len(coords):
        seeds[tuple(coords.T)] = True
    labels = s1.watershed(-land, s1.cc_label(seeds), mask=mask)
    sizes = np.bincount(labels.ravel()); keep = sizes >= cfg.min_object_voxels; keep[0] = False
    labels[np.isin(labels, np.nonzero(~keep)[0])] = 0
    ids = np.unique(labels); ids = ids[ids != 0]
    remap = np.zeros(int(labels.max()) + 1, labels.dtype); remap[ids] = np.arange(1, len(ids) + 1, dtype=labels.dtype)
    labels = remap[labels]
    accepted, _ = s1.classify_instances(
        labels, cfg.nucleus_diameter_vox, cfg.solidity_thresh, cfg.volume_zscore_thresh,
        cfg.min_volume_fraction_of_expected, cfg.max_volume_fraction_of_expected,
        accept_big_side_flags=getattr(cfg, "accept_big_side_flags", False))
    return accepted


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--z-range", default=None)
    ap.add_argument("--seed-distances", default="", help="optional extra min_seed_distance values to try with each variant")
    a = ap.parse_args()
    cfg = Stage1Config()
    raw = tifffile.imread(a.raw).astype(np.float32)
    gt = tifffile.imread(a.gt).astype(np.int32)
    z0, z1 = parse_z_range(a.z_range, gt.shape[0])

    variants = [("baseline", 0)] + [(k, s) for k in ("dt_smooth", "int_blend", "dt_from_smooth") for s in (0.5, 1.0, 1.5)]
    dists = [cfg.min_seed_distance] + [int(x) for x in a.seed_distances.split(",") if x]
    print(f"voxel_size={cfg.voxel_size}\n{'variant':>22}{'dist':>5}{'sec':>6}{'pred':>6}{'match':>7}{'extraFP':>8}"
          f"{'split':>6}{'merge':>6}{'prec':>7}{'recall':>8}{'F1':>7}{'IoU':>7}")
    from dataclasses import replace
    for d in dists:
        c = replace(cfg, min_seed_distance=d)
        for k, s in variants:
            t = time.perf_counter(); lab = segment(raw, c, k, s); dt = time.perf_counter() - t
            m = evaluate_one(lab[z0:z1], gt[z0:z1], 0.25, 0.15)
            print(f"{(k + ('=' + str(s) if s else '')):>22}{d:>5}{dt:>6.1f}{m['pred_count']:>6}{m['matched']:>7}"
                  f"{m['extra_false_positives']:>8}{m['split_gt_nuclei']:>6}{m['merged_pred_objects']:>6}"
                  f"{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['mean_matched_iou']:>7.3f}")

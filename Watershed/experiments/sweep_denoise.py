"""
Does a cheap pre-filter in front of stage 1 help? Scores stage 1 against ground truth after each
candidate denoise, and times each filter, so accuracy gain can be weighed against cost.

Filters (applied to the RAW chunk before run_stage1; sigmas are in physical units, converted to
voxels with config.voxel_size so the coarse axis isn't over-blurred):
  none
  gauss_um=S          3D Gaussian, sigma S micrometres on every axis
  gauss_inplane=S     Gaussian on the two fine axes only (coarse axis untouched)
  median3             3x3x3 median
  median_inplane      3x3 median in each plane of the fine axes (1x3x3)
  bgsub=S             subtract a very wide Gaussian (sigma S um) -- removes slow background, not noise
  gauss_inplane+bgsub combined

Usage:
    python experiments/sweep_denoise.py --raw data\\ground_truth\\gt_chunk1_raw.tif ^
        --gt data\\ground_truth\\ground_truth_chunk1.tif --z-range 1:127
"""
import argparse
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi

from nuclei_stage1.evaluate import evaluate_one, parse_z_range
from nuclei_stage1.config import Stage1Config
from nuclei_stage1.segmentation import run_stage1


def sig_vox(s_um, vs, axes=(0, 1, 2)):
    return tuple((s_um / vs[i]) if i in axes else 0.0 for i in range(3))


def make_filters(vs):
    f = {"none": lambda v: v}
    for s in (0.5, 1.0, 1.5):
        f[f"gauss_um={s}"] = lambda v, s=s: ndi.gaussian_filter(v, sig_vox(s, vs))
    for s in (0.5, 1.0, 1.5):
        f[f"gauss_inplane={s}"] = lambda v, s=s: ndi.gaussian_filter(v, sig_vox(s, vs, (1, 2)))
    f["median3"] = lambda v: ndi.median_filter(v, size=3)
    f["median_inplane"] = lambda v: ndi.median_filter(v, size=(1, 3, 3))
    f["bgsub=15"] = lambda v: np.clip(v - ndi.gaussian_filter(v, sig_vox(15, vs)), 0, None)
    f["gauss_inplane=1.0+bgsub=15"] = lambda v: np.clip(
        ndi.gaussian_filter(v, sig_vox(1.0, vs, (1, 2))) - ndi.gaussian_filter(v, sig_vox(15, vs)), 0, None)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--z-range", default=None)
    a = ap.parse_args()

    cfg = Stage1Config()
    raw = tifffile.imread(a.raw).astype(np.float32)
    gt = tifffile.imread(a.gt).astype(np.int32)
    z0, z1 = parse_z_range(a.z_range, gt.shape[0])

    t = time.perf_counter(); run_stage1(raw, cfg); t_stage1 = time.perf_counter() - t
    print(f"voxel_size={cfg.voxel_size}  stage 1 alone on this chunk: {t_stage1:.2f}s\n")
    print(f"{'filter':>28}{'filter s':>10}{'+% time':>9}{'pred':>6}{'match':>7}{'extraFP':>8}"
          f"{'split':>6}{'merge':>6}{'prec':>7}{'recall':>8}{'F1':>7}{'IoU':>7}")
    for name, fn in make_filters(cfg.voxel_size).items():
        t = time.perf_counter(); v = fn(raw); tf = time.perf_counter() - t
        m = evaluate_one(run_stage1(v, cfg)[0][z0:z1], gt[z0:z1], 0.25, 0.15)
        print(f"{name:>28}{tf:>10.2f}{100 * tf / t_stage1:>8.0f}%{m['pred_count']:>6}{m['matched']:>7}"
              f"{m['extra_false_positives']:>8}{m['split_gt_nuclei']:>6}{m['merged_pred_objects']:>6}"
              f"{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['mean_matched_iou']:>7.3f}")

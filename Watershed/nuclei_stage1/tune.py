"""
Tune stage 1 on ONE human-corrected ground-truth chunk.

Give it a corrected chunk and it grid-searches the parameters that mattered in our tuning, so you do not
hand-edit config.py for every new region.

It grid-searches target_foreground_percentile x min_seed_distance x seed_landscape_smooth_um x mask type (raw per-slice
or band-pass; everything else stays at config.py), scores each against the GT with the same metrics as evaluate.py,
ranks by F1 (ties -> mean IoU), prints the top results, and prints the best setting.

Caveats, because this is only as good as the labels behind it:
  * one chunk's best-F1 is partly overfitting; look at how FLAT the top of the table is. If several settings
    are within ~0.01 F1, any of them is as good, and --pick-conservative takes the one with the fewest
    extra false positives among those.
  * a setting tuned on one tissue region may not transfer to another; tune per region if you have the labels.

Usage:
    python -m nuclei_stage1.tune --raw data/ground_truth/gt_chunk1_raw.tif --gt data/ground_truth/ground_truth_chunk1.tif \
        --z-range 1:127 --name pfc_chunk1 --workers 8
    ... --percentiles 90,94,96 --seed-distances 10,12,17 --smooth 0,1.0
    ... --bandpass none,2:8,1:8           # also try the band-pass mask (percentiles ~92-96 suit it)
    ... --register                        # store the winner as an example for automatic selection (autoconfig.py)
"""
import argparse
import itertools
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import tifffile

from .evaluate import evaluate_one, parse_z_range
from .config import Stage1Config
from .segmentation import run_stage1

_G = {}


def _init(raw, gt, z0, z1, base):
    _G.update(raw=raw, gt=gt, z0=z0, z1=z1, base=base)


def _run(params):
    p, d, s, bp = params
    cfg = replace(_G["base"], target_foreground_percentile=float(p), min_seed_distance=int(d),
                  seed_landscape_smooth_um=float(s), mask_bandpass_um=bp)
    lab = run_stage1(_G["raw"], cfg)[0]
    m = evaluate_one(lab[_G["z0"]:_G["z1"]], _G["gt"][_G["z0"]:_G["z1"]], 0.25, 0.15)
    return params, m


def floats(s): return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--z-range", default=None); ap.add_argument("--name", required=True)
    ap.add_argument("--percentiles", default="88,90,92,94,96")
    ap.add_argument("--seed-distances", default="10,12,14,17")
    ap.add_argument("--smooth", default="0,1.0")
    ap.add_argument("--bandpass", default="none",
                    help="mask options to try, comma-separated: 'none' (raw per-slice mask) or lo:hi um, e.g. none,2:8,1:8")
    ap.add_argument("--register", action="store_true",
                    help="store the chosen setting + this chunk's image statistics as an example (autoconfig.py)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pick-conservative", action="store_true",
                    help="among settings within 0.01 F1 of the best, take the fewest extra false positives")
    a = ap.parse_args()

    base = Stage1Config()
    raw = tifffile.imread(a.raw); gt = tifffile.imread(a.gt).astype(np.int32)
    z0, z1 = parse_z_range(a.z_range, gt.shape[0])
    bps = [None if b == "none" else tuple(float(x) for x in b.split(":")) for b in a.bandpass.split(",")]
    grid = list(itertools.product(floats(a.percentiles), [int(x) for x in floats(a.seed_distances)], floats(a.smooth), bps))
    print(f"{len(grid)} settings on {a.raw}, z=[{z0}:{z1}], {int((gt[z0:z1] > 0).any() and len(np.unique(gt[z0:z1])) - 1)} GT nuclei")
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(raw, gt, z0, z1, base)) as ex:
        res = list(ex.map(_run, grid, chunksize=1))
    res.sort(key=lambda r: (-r[1]["f1"], -r[1]["mean_matched_iou"]))
    print(f"{'pct':>5}{'seed':>6}{'smooth':>7}{'band':>8}{'pred':>6}{'match':>7}{'extraFP':>8}{'split':>6}{'prec':>7}{'recall':>8}{'F1':>7}{'IoU':>7}")
    for (p, d, s, bp), m in res[:10]:
        print(f"{p:>5g}{d:>6}{s:>7g}{('-' if bp is None else '%g:%g' % bp):>8}{m['pred_count']:>6}{m['matched']:>7}{m['extra_false_positives']:>8}"
              f"{m['split_gt_nuclei']:>6}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['mean_matched_iou']:>7.3f}")
    best = res[0]
    if a.pick_conservative:
        near = [r for r in res if r[1]["f1"] >= best[1]["f1"] - 0.01]
        best = min(near, key=lambda r: (r[1]["extra_false_positives"], -r[1]["f1"]))
    (p, d, s, bp), m = best
    n_flat = sum(r[1]["f1"] >= res[0][1]["f1"] - 0.01 for r in res)
    print(f"\nchosen: percentile={p:g} min_seed_distance={d} seed_landscape_smooth_um={s:g} bandpass={bp}  "
          f"(F1 {m['f1']:.3f}; {n_flat} of {len(res)} settings are within 0.01 F1 of the best -- "
          f"{'a flat top, so treat the choice as soft' if n_flat > 3 else 'a fairly sharp optimum'})")
    if a.register:
        from .autoconfig import add_exemplar
        ov = {"target_foreground_percentile": float(p), "min_seed_distance": int(d), "seed_landscape_smooth_um": float(s)}
        if bp:
            ov["mask_bandpass_um"] = bp
        n = add_exemplar(raw, a.name, ov, tuned_f1=round(m["f1"], 3), gt=Path(a.gt).name)
        print(f"registered '{a.name}' ({n} example(s) total) -> exemplars.json")


if __name__ == "__main__":
    main()

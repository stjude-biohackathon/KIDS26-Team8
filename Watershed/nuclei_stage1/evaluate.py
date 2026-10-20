"""
Compare any number of named prediction label volumes (watershed,
stardist, cellpose_sam, ...) against a real ground-truth label volume,
using globally-optimal Hungarian matching (not greedy nearest-centroid,
which isn't guaranteed optimal) -- adapted from the teammate's
04_evaluate.py, extended to:
  - handle MULTIPLE predictions in one run, printed as a single
    side-by-side table
  - restrict comparison to a Z-RANGE you've actually finished
    annotating, so an incomplete ground truth chunk doesn't get
    silently compared against its own un-annotated (stage-1 leftover)
    portion, which would corrupt the numbers

Usage:
    python -m nuclei_stage1.evaluate \
        --gt ground_truth_chunk1.tif \
        --pred watershed=chunk1_stage1_labels.tif \
        --pred stardist=chunk1_stardist_labels.tif \
        --pred cellpose_sam=chunk1_cellpose_labels.tif

    python -m nuclei_stage1.evaluate \
        --gt ground_truth_chunk1.tif --z-range 40:80 \
        --pred watershed=chunk1_stage1_labels.tif \
        --iou-threshold 0.25
"""
import argparse
import sys

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


def parse_z_range(spec, depth):
    if spec is None:
        return 0, depth
    lo, hi = spec.split(":")
    lo = int(lo) if lo else 0
    hi = int(hi) if hi else depth
    if not (0 <= lo < hi <= depth):
        raise ValueError(f"--z-range {spec} is out of bounds for a volume with depth {depth}")
    return lo, hi


def _drop_small(lab, min_voxels):
    """Zero out instances with fewer than min_voxels voxels INSIDE THIS WINDOW. An
    instance cut by the edge of the evaluated z-range leaves a tiny sliver that no
    method can be expected to match; if both a ground-truth sliver and a prediction
    sliver land in the same spot they count as a 'match', and if only one exists it
    counts as a miss or a false positive. Either way it's noise, not detection quality."""
    if min_voxels <= 0:
        return lab, 0
    counts = np.bincount(lab.ravel())
    small = np.where(counts < min_voxels)[0]
    small = small[(small != 0) & (counts[small] > 0)]
    if len(small):
        lab = lab.copy()
        lab[np.isin(lab, small)] = 0
    return lab, int(len(small))


def evaluate_one(pred, gt, iou_threshold, min_contribution=0.15, min_voxels=0):
    """Adapted from the teammate's 04_evaluate.py: Hungarian-matched
    precision/recall/F1/matched-IoU stats (unchanged, still IoU-based --
    that's a separate, standard "best 1-1 matching" question). split_gt_nuclei
    and merged_pred_objects use CONTRIBUTION instead of IoU-threshold edges
    (see docs/METHODS.md) -- IoU-threshold splitting/merging counts
    silently miss a real split/merge whenever a nucleus is fragmented into
    3+ roughly-equal pieces, since each piece's IoU against the whole
    nucleus (dividing by the union, dominated by the full nucleus/prediction
    size) can land below the threshold even though the piece clearly,
    substantially belongs to that nucleus. min_contribution instead asks
    "does this piece account for at least this fraction of the OTHER
    side's volume" -- directly catches multi-way splits/merges regardless
    of how many pieces they're divided into."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} vs gt={gt.shape}")
    pred, n_pred_ignored = _drop_small(pred, min_voxels)
    gt, n_gt_ignored = _drop_small(gt, min_voxels)

    pids = np.unique(pred); pids = pids[pids > 0]
    gids = np.unique(gt); gids = gids[gids > 0]

    P = int(pred.max()) + 1
    mask = (pred > 0) & (gt > 0)
    keys = gt[mask].astype(np.int64) * P + pred[mask].astype(np.int64)
    keys, cnt = np.unique(keys, return_counts=True)
    pa = np.bincount(pred.ravel(), minlength=P)
    ga = np.bincount(gt.ravel(), minlength=int(gt.max()) + 1)

    edges = []
    contributes_to_gt = {}   # g -> list of q's contributing >=min_contribution of g's volume
    contributes_to_pred = {}  # q -> list of g's contributing >=min_contribution of q's volume
    for k, c in zip(keys, cnt):
        g = int(k // P); q = int(k % P)
        u = ga[g] + pa[q] - int(c)
        i = float(c / u) if u else 0
        if i >= iou_threshold:
            edges.append((g, q, i))
        if ga[g] and (c / ga[g]) >= min_contribution:
            contributes_to_gt.setdefault(g, []).append(q)
        if pa[q] and (c / pa[q]) >= min_contribution:
            contributes_to_pred.setdefault(q, []).append(g)

    gs = sorted({e[0] for e in edges}); ps = sorted({e[1] for e in edges})
    matches = []
    if edges:
        gi = {g: i for i, g in enumerate(gs)}; pi = {q: i for i, q in enumerate(ps)}
        cost = np.ones((len(gs), len(ps)), np.float32)
        for g, q, i in edges:
            cost[gi[g], pi[q]] = 1 - i
        rr, cc = linear_sum_assignment(cost)
        matches = [(gs[r], ps[c], 1 - float(cost[r, c])) for r, c in zip(rr, cc)
                   if 1 - float(cost[r, c]) >= iou_threshold]

    mg = {g for g, _, _ in matches}; mp = {q for _, q, _ in matches}
    split = sum(len(contributes_to_gt.get(int(g), [])) >= 2 for g in gids)
    merge = sum(len(contributes_to_pred.get(int(q), [])) >= 2 for q in pids)
    tp = len(matches); ng = len(gids); npred = len(pids)
    prec = tp / npred if npred else 0
    rec = tp / ng if ng else 0

    return {
        "pred_count": npred, "gt_count": ng, "matched": tp,
        "missed_false_negatives": ng - len(mg), "extra_false_positives": npred - len(mp),
        "ignored_tiny_gt": n_gt_ignored, "ignored_tiny_pred": n_pred_ignored,
        "split_gt_nuclei": int(split), "merged_pred_objects": int(merge),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
        "count_error": npred - ng, "absolute_count_error": abs(npred - ng),
        "mean_matched_iou": float(np.mean([m[2] for m in matches])) if matches else 0,
        "median_matched_iou": float(np.median([m[2] for m in matches])) if matches else 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="path to the ground truth label TIFF")
    p.add_argument("--pred", action="append", required=True,
                    help="name=path, repeatable -- e.g. --pred watershed=... --pred stardist=...")
    p.add_argument("--z-range", default=None,
                    help="lo:hi -- restrict comparison to the z-slices you've actually "
                         "finished annotating. Omit to use the whole volume (only correct "
                         "if the ENTIRE chunk is fully annotated).")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--ignore-below-voxels", type=int, default=0,
                    help="ignore ground-truth AND predicted instances with fewer than this many "
                         "voxels inside the evaluated window (default 0 = off). These are "
                         "slivers of nuclei cut by the z-range edge; try 100")
    p.add_argument("--min-contribution", type=float, default=0.15,
                    help="fraction of the OTHER side's volume a piece must account for to "
                         "count toward split_gt_nuclei/merged_pred_objects (default 0.15) -- "
                         "see docs/METHODS.md")
    a = p.parse_args()

    gt_full = tifffile.imread(a.gt).astype(np.int32)
    z0, z1 = parse_z_range(a.z_range, gt_full.shape[0])
    gt = gt_full[z0:z1]
    n_gt_instances = len(np.unique(gt)) - (1 if 0 in gt else 0)

    print(f"ground truth: {a.gt}  full_shape={gt_full.shape}  "
          f"comparing z=[{z0}:{z1}] only ({z1-z0}/{gt_full.shape[0]} slices)  "
          f"n_instances_in_range={n_gt_instances}")
    if a.z_range is None:
        print("  NOTE: no --z-range given -- assuming the WHOLE volume is fully annotated. "
              "If it isn't, results below are meaningless (comparing against un-annotated "
              "leftover labels, not real ground truth).")
    print()

    results = {}
    pred_paths = {}
    for spec in a.pred:
        if "=" not in spec:
            print(f"ERROR: --pred must be name=path, got: {spec}")
            sys.exit(1)
        name, path = spec.split("=", 1)
        pred_paths[name] = path
        pred_full = tifffile.imread(path).astype(np.int32)
        if pred_full.shape != gt_full.shape:
            print(f"ERROR: {name} shape {pred_full.shape} doesn't match gt shape {gt_full.shape}")
            sys.exit(1)
        pred = pred_full[z0:z1]
        results[name] = evaluate_one(pred, gt, a.iou_threshold, a.min_contribution,
                                     a.ignore_below_voxels)

    cols = list(results.keys())
    rows = ["pred_count", "gt_count", "matched", "missed_false_negatives",
            "extra_false_positives", "split_gt_nuclei", "merged_pred_objects",
            "precision", "recall", "f1", "count_error", "mean_matched_iou", "median_matched_iou"]
    if a.ignore_below_voxels > 0:
        rows = ["ignored_tiny_gt", "ignored_tiny_pred"] + rows

    header = f"{'metric':>26}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = []
        for c in cols:
            v = results[c][row]
            vals.append(f"{v:>16.3f}" if isinstance(v, float) else f"{v:>16}")
        print(f"{row:>26}" + "".join(vals))

    print(f"\niou_threshold={a.iou_threshold}  (a pred/gt pair counts as matched only if "
          f"IoU >= this, for precision/recall/F1/matched-IoU)")
    print(f"min_contribution={a.min_contribution}  (fraction of the OTHER side's volume "
          f"needed to count toward split_gt_nuclei/merged_pred_objects)")


if __name__ == "__main__":
    main()

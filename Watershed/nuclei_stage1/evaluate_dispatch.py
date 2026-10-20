"""
Does descriptor-driven config selection (autoconfig.py) actually beat
one global config on chunks it has never seen?  Leave-one-chunk-out test.

For every GT chunk in the manifest:
  1. GLOBAL    -- stage 1 with the global PipelineConfig defaults.
  2. ORACLE    -- best config found by grid search ON THAT CHUNK. An optimistic
                  upper bound (tuned on the thing it's scored on), not a result.
  3. DISPATCH  -- config chosen by autoconfig.select_config using ONLY the
                  other chunks' oracle configs as exemplars. The chunk being
                  scored is never seen. THIS is the number that matters.
  4. SHUFFLE   -- mean F1 over each other chunk's oracle config applied blindly.
                  Control: if DISPATCH can't beat this, the descriptors aren't
                  adding anything beyond "any tuned config is better than none".

"gap closed" = (DISPATCH - GLOBAL) / (ORACLE - GLOBAL): the share of the
available improvement the dispatcher captured without seeing the chunk.

HONEST LIMIT: if a held-out chunk has no exemplar of a similar density (e.g.
your only hippocampus chunk is held out and the rest are cortex), the
dispatcher SHOULD fall back to global -- the test then reports a fallback and
tells you nothing about whether matching works. You need >= 2 labeled chunks
per density regime for a meaningful result.

Manifest (JSON list):
  [{"name": "pfc1",   "raw": "gt_chunk1_raw.tif", "gt": "ground_truth_chunk1.tif", "z_range": "1:127"},
   {"name": "hippo1", "raw": "gt_chunk3_raw.tif", "gt": "ground_truth_chunk3.tif", "z_range": "1:127"}]

Usage:
    python -m nuclei_stage1.evaluate_dispatch --manifest gt_manifest.json
    python -m nuclei_stage1.evaluate_dispatch --manifest gt_manifest.json --seed-distances 12,17,22 \
        --seed-prominences 0.1,0.2 --percentiles 88,90,92
"""
import argparse
import itertools
import json
from dataclasses import replace

import numpy as np
import tifffile

from .evaluate import evaluate_one, parse_z_range
from .descriptors import describe_chunk
from .config import Stage1Config
from .autoconfig import match_vector as _vec, select_config
from .segmentation import run_stage1


def f1_for(raw, gt, z0, z1, cfg, cache, key):
    if key not in cache:
        accepted = run_stage1(raw, cfg)[0]
        m = evaluate_one(accepted[z0:z1], gt[z0:z1], 0.25, 0.15)
        cache[key] = (m["f1"], m["matched"], m["extra_false_positives"])
    return cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed-distances", default="12,17")
    ap.add_argument("--seed-prominences", default="0.15")
    ap.add_argument("--percentiles", default="90,94")
    ap.add_argument("--bandpass", default="none,2:8,1:8",
                    help="mask options per config: 'none' (raw per-slice) or lo:hi um difference-of-Gaussians (default none,2:8,1:8)")
    ap.add_argument("--smooth", type=float, default=1.0, help="seed_landscape_smooth_um used for every grid config")
    ap.add_argument("--max-distance", type=float, default=0.75)
    a = ap.parse_args()

    bps = [None if b == "none" else tuple(float(x) for x in b.split(":")) for b in a.bandpass.split(",")]
    grid = list(itertools.product([int(v) for v in a.seed_distances.split(",")],
                                  [float(v) for v in a.seed_prominences.split(",")],
                                  [float(v) for v in a.percentiles.split(",")], bps))
    base = Stage1Config()
    chunks, cache = [], {}
    for entry in json.load(open(a.manifest)):
        raw = tifffile.imread(entry["raw"])
        gt = tifffile.imread(entry["gt"]).astype(np.int32)
        z0, z1 = parse_z_range(entry.get("z_range"), gt.shape[0])
        chunks.append({"name": entry["name"], "raw": raw, "gt": gt, "z0": z0, "z1": z1,
                       "desc": describe_chunk(raw, base.nucleus_diameter_vox)})
    if len(chunks) < 2:
        raise SystemExit("need at least 2 labeled chunks for a leave-one-out test")

    print(f"grid: {len(grid)} configs per chunk x {len(chunks)} chunks\n")
    for c in chunks:                                    # oracle search per chunk
        best = None
        for sd, sp, pct, bp in grid:
            cfg = replace(base, min_seed_distance=sd, seed_prominence=sp,
                          target_foreground_percentile=pct, mask_bandpass_um=bp, seed_landscape_smooth_um=a.smooth)
            f1, matched, _ = f1_for(c["raw"], c["gt"], c["z0"], c["z1"], cfg, cache,
                                    (c["name"], sd, sp, pct, bp))
            if best is None or (f1, matched) > best[0]:
                best = ((f1, matched), {"min_seed_distance": sd, "seed_prominence": sp,
                                        "target_foreground_percentile": pct, "seed_landscape_smooth_um": a.smooth,
                                        **({"mask_bandpass_um": list(bp)} if bp else {})})
        c["oracle_f1"], c["oracle_overrides"] = best[0][0], best[1]
        c["global_f1"] = f1_for(c["raw"], c["gt"], c["z0"], c["z1"], base, cache, (c["name"], "global"))[0]
        print(f"{c['name']:<12} global F1={c['global_f1']:.3f}  oracle F1={c['oracle_f1']:.3f}  "
              f"oracle={c['oracle_overrides']}")

    print(f"\n{'held-out':<12}{'global':>8}{'dispatch':>10}{'shuffle':>9}{'oracle':>8}"
          f"{'gap closed':>12}   dispatch source")
    gaps = []
    for i, c in enumerate(chunks):
        others = [o for j, o in enumerate(chunks) if j != i]
        exemplars = [{"name": o["name"], "overrides": o["oracle_overrides"],
                      "match_vec": _vec(o["desc"]).tolist()} for o in others]
        cfg_d, info = select_config(c["raw"], exemplars=exemplars, max_distance=a.max_distance, base=base)
        f1_d = f1_for(c["raw"], c["gt"], c["z0"], c["z1"], cfg_d, cache,
                      (c["name"], "dispatch", repr(sorted(info.get("overrides", {}).items()))))[0]
        f1_s = float(np.mean([
            f1_for(c["raw"], c["gt"], c["z0"], c["z1"], replace(base, **o["oracle_overrides"]), cache,
                   (c["name"], "shuffle", o["name"]))[0] for o in others]))
        room = c["oracle_f1"] - c["global_f1"]
        gap = (f1_d - c["global_f1"]) / room if room > 1e-9 else float("nan")
        if info["source"].startswith("exemplar"):
            gaps.append(gap)
        print(f"{c['name']:<12}{c['global_f1']:>8.3f}{f1_d:>10.3f}{f1_s:>9.3f}{c['oracle_f1']:>8.3f}"
              f"{gap:>12.0%}   {info['source']}" + ("" if info['source'].startswith('exemplar')
                                                    else "  <- fell back: no similar exemplar, uninformative"))

    print()
    if not gaps:
        print("Every held-out chunk fell back to global -- no chunk had a close-enough exemplar, so "
              "this run cannot say whether matching works. Add labeled chunks of similar density.")
    else:
        print(f"{len(gaps)}/{len(chunks)} held-out chunks were matched to an exemplar; mean gap closed "
              f"(matched ones only) = {np.nanmean(gaps):.0%}. Read DISPATCH against SHUFFLE: dispatch must "
              f"beat shuffle for descriptors to be doing real work. With few chunks this is "
              f"anecdote, not a measurement.")


if __name__ == "__main__":
    main()

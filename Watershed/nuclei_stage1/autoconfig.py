"""
Automatic per-chunk parameter selection. EXPERIMENTAL -- a working framework, not a validated method.
See docs/AUTO_CONFIG.md for the full story and its limits.

What it does, in one paragraph: it measures a few cheap image statistics of a chunk (descriptors.py), finds the
nearest *registered example* (a corrected ground-truth chunk plus the settings that were tuned on it), and applies that
example's settings. If no example is close enough the chunk gets the global defaults and says so. Nothing is fitted and
no labels are needed at run time -- but the examples themselves come from corrected chunks, so the scheme can only
recognise tissue that looks like something you already corrected.

Examples live in exemplars.json next to this file (two so far: PFC chunk with the validated defaults, hippocampus chunk
with the `hippocampus` preset).

Usage:
    python -m nuclei_stage1.autoconfig select chunk.tif              # what would this chunk get?
    python -m nuclei_stage1.autoconfig survey <slab_dir> --n 40      # dry run over a slab (no segmentation)
    python -m nuclei_stage1.autoconfig register gt_raw.tif --name hippo2 \\
        --set mask_bandpass_um=2:8 --set target_foreground_percentile=94 --set min_seed_distance=12
    # (nuclei_stage1.tune --register does the tuning AND the registering in one step)
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile

from .config import Stage1Config
from .descriptors import describe_chunk

EXEMPLAR_PATH = Path(__file__).with_name("exemplars.json")

# descriptors used for matching (all positive, compared in log space)
MATCH_KEYS = ("fg_frac_k3", "blobs_per_mvox", "contrast_p99")

# only these may be overridden per chunk -- geometry / IO settings stay global
TUNABLE = {
    "min_seed_distance": int, "seed_prominence": float,
    "target_foreground_percentile": float, "solidity_thresh": float,
    "volume_zscore_thresh": float, "min_volume_fraction_of_expected": float,
    "seed_landscape_smooth_um": float, "background_floor_sigma": float, "min_object_voxels": int,
    # "lo:hi" um difference-of-Gaussians mask; omit it (or None) for the default raw per-slice mask
    "mask_bandpass_um": lambda v: tuple(float(x) for x in (v.replace(":", ",").split(",") if isinstance(v, str) else v)),
}


def match_vector(desc):
    return np.log(np.array([max(desc[k], 1e-6) for k in MATCH_KEYS]))


def load_exemplars(path=EXEMPLAR_PATH):
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def add_exemplar(raw, name, overrides, tuned_f1=None, gt=None, path=EXEMPLAR_PATH):
    """Register one corrected chunk: its descriptors plus the settings tuned on it. Re-registering a name replaces it."""
    desc = describe_chunk(raw)
    if not desc.get("usable"):
        raise ValueError("chunk is mostly zero padding; not a usable example")
    ov = {k: (list(v) if isinstance(v, tuple) else v) for k, v in overrides.items()}
    ex = [e for e in load_exemplars(path) if e["name"] != name]
    ex.append({"name": name, "descriptors": desc, "match_vec": match_vector(desc).tolist(), "overrides": ov,
               **({"tuned_f1": tuned_f1} if tuned_f1 is not None else {}), **({"gt": gt} if gt else {})})
    Path(path).write_text(json.dumps(ex, indent=2))
    return len(ex)


def select_config(vol, exemplars=None, max_distance=0.75, base=None):
    """Returns (Stage1Config, info). info['source'] is 'exemplar:<name>' or 'global-default', with the reason, so every
    run can log which it was.

    A chunk matched to an example gets exactly that example's overrides, and any setting the example does not list is
    taken from `base`; the band-pass mask is explicitly switched OFF unless the example lists it, so a plain example
    (e.g. PFC) is never contaminated by a preset on `base`."""
    base = base or Stage1Config()
    exemplars = load_exemplars() if exemplars is None else exemplars
    desc = describe_chunk(vol, base.nucleus_diameter_vox)
    if not desc.get("usable"):
        return base, {"source": "global-default", "reason": "chunk mostly zero padding", "descriptors": desc}
    if not exemplars:
        return base, {"source": "global-default", "reason": "no examples registered", "descriptors": desc}

    v = match_vector(desc)
    dists = [float(np.linalg.norm(v - np.array(e["match_vec"]))) for e in exemplars]
    i = int(np.argmin(dists))
    info = {"descriptors": desc, "nearest": exemplars[i]["name"], "distance": dists[i]}
    if dists[i] > max_distance:
        info.update(source="global-default",
                    reason=f"nearest example '{exemplars[i]['name']}' is {dists[i]:.2f} away (> {max_distance}) "
                           f"-- chunk unlike anything tuned on")
        return base, info
    ov = {k: (tuple(x) if isinstance(x, list) else x) for k, x in exemplars[i]["overrides"].items()}
    ov.setdefault("mask_bandpass_um", None)
    info.update(source=f"exemplar:{exemplars[i]['name']}", reason="within match distance", overrides=ov)
    return replace(base, **ov), info


def _parse_sets(pairs):
    out = {}
    for s in pairs:
        k, v = s.split("=", 1)
        if k not in TUNABLE:
            sys.exit(f"'{k}' is not overridable; choose from {sorted(TUNABLE)}")
        out[k] = TUNABLE[k](v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("raw"); r.add_argument("--name", required=True)
    r.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE")
    s = sub.add_parser("select"); s.add_argument("raw"); s.add_argument("--max-distance", type=float, default=0.75)
    v = sub.add_parser("survey", help="dry run: which setting would each sampled chunk of a slab get? (no segmentation)")
    v.add_argument("slab_dir"); v.add_argument("--n", type=int, default=40, help="chunks to sample (evenly spaced)")
    v.add_argument("--max-distance", type=float, default=0.75)
    a = ap.parse_args()

    if a.cmd == "survey":
        from .chunks import list_chunks
        chunks = list_chunks(a.slab_dir)
        pick = chunks[:: max(1, len(chunks) // a.n)][: a.n]
        tally = Counter()
        for c in pick:
            _, info = select_config(tifffile.imread(c.raw_tif), max_distance=a.max_distance)
            tally[info["source"]] += 1
            print(f"  {c.name:<44}{info['source']:<28}dist={info.get('distance', float('nan')):.2f}")
        print(f"\n{len(pick)} of {len(chunks)} chunks sampled: " + ", ".join(f"{k}: {n}" for k, n in tally.most_common()))
        print("Many 'global-default' fallbacks = tissue unlike every registered example -> correct a chunk of it.")
        return

    vol = tifffile.imread(a.raw)
    if a.cmd == "register":
        n = add_exemplar(vol, a.name, _parse_sets(a.set))
        print(f"registered '{a.name}' ({n} example(s) total) -> {EXEMPLAR_PATH.name}")
    else:
        cfg, info = select_config(vol, max_distance=a.max_distance)
        print(f"source: {info['source']}  ({info['reason']})")
        if "nearest" in info:
            print(f"nearest example: {info['nearest']}  distance={info['distance']:.2f}")
        if info.get("descriptors", {}).get("usable"):
            print("descriptors:", {k: round(x, 3) for k, x in info["descriptors"].items() if k in MATCH_KEYS})
        print("overrides applied:", info.get("overrides", {}) or "none")


if __name__ == "__main__":
    main()

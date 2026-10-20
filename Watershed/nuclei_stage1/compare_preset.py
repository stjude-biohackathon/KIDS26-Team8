"""
Run stage 1 with the default settings and with a named preset (config.PRESETS) on one corrected chunk, save both label
volumes, and print the side-by-side score (matched, false positives, splits, merges, F1, IoU).

Usage:
    python -m nuclei_stage1.compare_preset --raw gt_raw.tif --gt gt.tif --preset hippocampus [--z-range 1:127] [--out outputs]
"""
import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile

from .config import PRESETS, Stage1Config
from .segmentation import run_stage1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--preset", default="hippocampus", choices=sorted(PRESETS))
    ap.add_argument("--z-range", default=None, help="lo:hi, only the slices you finished correcting")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    raw = tifffile.imread(a.raw)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.gt).stem
    preds = []
    for name, cfg in (("default", Stage1Config()), (a.preset, replace(Stage1Config(), **PRESETS[a.preset]))):
        p = out / f"{stem}_{name}_labels.tif"
        tifffile.imwrite(p, run_stage1(raw, cfg)[0].astype(np.uint32))
        preds += ["--pred", f"{name}={p}"]
        print("wrote", p)
    cmd = [sys.executable, "-m", "nuclei_stage1.evaluate", "--gt", a.gt] + preds + (["--z-range", a.z_range] if a.z_range else [])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""End to end on a fake slab: a nucleus sitting on a tile boundary must come out as ONE object after stitching
(and with the experimental overlapping-edge run), and as two pieces with plain per-tile processing."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from helpers import make_slab

REPO = Path(__file__).resolve().parent.parent
NUCLEUS = [((60, 128, 60), (3, 9, 9))]          # centred exactly on the y=128 tile boundary


def run(module, slab, *args):
    r = subprocess.run([sys.executable, "-m", module, str(slab), *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def ids_per_tile(folder):
    out = {}
    for p in sorted(Path(folder).glob("*_labels.tif")):
        lab = tifffile.imread(p)
        out[p.name] = set(np.unique(lab[lab > 0]).tolist())
    return out


def test_plain_run_cuts_the_nucleus(tmp_path):
    slab = make_slab(tmp_path / "s", NUCLEUS)
    run("nuclei_stage1.run_slab", slab)
    per = ids_per_tile(slab / "labels")
    assert all(len(v) == 1 for v in per.values())               # one piece in each tile
    assert len(pd.concat([pd.read_csv(p) for p in (slab / "centroids").glob("*.csv")])) == 2


def test_stitch_merges_the_pieces(tmp_path):
    slab = make_slab(tmp_path / "s", NUCLEUS)
    run("nuclei_stage1.run_slab", slab)
    run("nuclei_stage1.stitch", slab)
    per = ids_per_tile(slab / "labels_stitched")
    union = set().union(*per.values())
    assert len(union) == 1 and all(v == union for v in per.values())
    assert len(pd.read_csv(slab / "labels_stitched" / "stitched_centroids.csv")) == 1


def test_halo_keeps_the_nucleus_whole_and_counts_it_once(tmp_path):
    slab = make_slab(tmp_path / "s", NUCLEUS)
    run("nuclei_stage1.experimental.run_halo", slab)
    per = ids_per_tile(slab / "labels_merged")
    union = set().union(*per.values())
    assert len(union) == 1 and all(v == union for v in per.values())      # same id on both sides of the boundary
    cents = pd.concat([pd.read_csv(p) for p in (slab / "centroids").glob("*.csv")])
    assert len(cents) == 1                                                # owned by exactly one tile
    assert abs(cents.iloc[0]["y"] - 128) < 4
    assert (slab / "labels_merged" / ".global_ids").exists()


def test_halo_resume_skips_finished_tiles(tmp_path):
    slab = make_slab(tmp_path / "s", NUCLEUS)
    run("nuclei_stage1.experimental.run_halo", slab)
    out = run("nuclei_stage1.experimental.run_halo", slab, "--skip-existing")
    assert "0 to segment" in out

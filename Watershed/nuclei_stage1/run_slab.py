"""
Run stage 1 on every tile of a slab, one tile at a time (no overlap between tiles).

For each <slab>/raw_chunked/{name}.tif writes
    <slab>/labels/{name}_labels.tif       uint32 label volume (ids are per tile, 1..n)
    <slab>/centroids/{name}_centroids.csv label_id, z, y, x, voxel_count  (ROI voxel coordinates, native frame)

A nucleus cut by a tile boundary comes out as separate labels in each tile. Either follow with
`python -m nuclei_stage1.stitch <slab>` to merge the pieces afterwards -- see README.

Usage:
    python -m nuclei_stage1.run_slab /path/to/slab --workers 8
    python -m nuclei_stage1.run_slab /path/to/slab --workers 8 --skip-existing     # resume after a time-out
    python -m nuclei_stage1.run_slab /path/to/slab --deskew                        # optional, see docs/METHODS.md
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from .chunks import list_chunks
from .config import PRESETS, Stage1Config
from .deskew import deskew_volume, to_native_points
from .segmentation import run_stage1


def process_chunk(chunk, cfg, deskew=False, adaptive=False, max_distance=0.75):
    """Returns (accepted_labels, difficult_mask, native_shape, used_cfg, info). native_shape is None unless deskewed.
    adaptive=True picks the settings per tile from its image statistics (autoconfig.select_config; experimental)."""
    raw = tifffile.imread(chunk.raw_tif)
    native_shape, valid = None, None
    if deskew:
        native_shape = raw.shape
        raw, valid = deskew_volume(raw)       # `valid` marks real voxels so the padding wedge is never data
    info = {"source": "global-default", "reason": "automatic selection off"}
    if adaptive:
        from .autoconfig import select_config
        cfg, info = select_config(raw, max_distance=max_distance, base=cfg)
    accepted, difficult, _ = run_stage1(raw, cfg, valid=valid)
    if valid is not None:
        accepted = accepted.copy(); accepted[~valid] = 0
    return accepted, difficult, native_shape, cfg, info


def write_outputs(chunk, labels, slab_dir, native_shape=None):
    """If labels are in the deskewed frame (native_shape set) the TIFF is saved as *_labels_deskewed.tif and
    centroids are mapped back to native chunk coordinates before the tile offset is added."""
    labels_dir = Path(slab_dir) / "labels"; centroids_dir = Path(slab_dir) / "centroids"
    labels_dir.mkdir(exist_ok=True); centroids_dir.mkdir(exist_ok=True)
    suffix = "_labels_deskewed.tif" if native_shape is not None else "_labels.tif"
    tifffile.imwrite(labels_dir / f"{chunk.name}{suffix}", labels.astype(np.uint32))
    rows = []
    for p in regionprops(labels):
        cz, cy, cx = p.centroid
        if native_shape is not None:
            cz, cy, cx = to_native_points([cz, cy, cx], native_shape)[0]
        rows.append({"label_id": p.label, "z": cz + chunk.z, "y": cy + chunk.y, "x": cx + chunk.x,
                     "voxel_count": p.area})
    pd.DataFrame(rows, columns=["label_id", "z", "y", "x", "voxel_count"]).to_csv(
        centroids_dir / f"{chunk.name}_centroids.csv", index=False)
    return len(rows)


def _run_one(job):
    chunk, cfg, slab_dir, deskew, adaptive, max_distance = job
    t0 = time.perf_counter()
    labels, difficult, native_shape, used, info = process_chunk(chunk, cfg, deskew, adaptive, max_distance)
    n = write_outputs(chunk, labels, slab_dir, native_shape)
    row = {"chunk": chunk.name, "seconds": round(time.perf_counter() - t0, 2), "source": info["source"], "reason": info["reason"], "nearest": info.get("nearest", ""),
           "distance": info.get("distance", ""), "mask_bandpass_um": str(used.mask_bandpass_um),
           "target_foreground_percentile": used.target_foreground_percentile, "min_seed_distance": used.min_seed_distance,
           "seed_landscape_smooth_um": used.seed_landscape_smooth_um, "n_nuclei": n,
           **{f"desc_{k}": v for k, v in info.get("descriptors", {}).items() if isinstance(v, (int, float))}}
    return f"  {chunk.name}: {n} nuclei, {difficult.mean():.4f} of voxels flagged [{info['source']}]", row


def run_on_slab(slab_dir, cfg, chunk_names=None, deskew=False, workers=1, skip_existing=False,
                adaptive=False, max_distance=0.75):
    """chunk_names: optional iterable of tile names to restrict to. workers > 1 runs tiles in parallel
    processes; tiles are independent, so results equal a serial run. skip_existing skips tiles whose centroid
    CSV already exists (resume after a time-out)."""
    chunks = list_chunks(slab_dir)
    if chunk_names is not None:
        chunks = [c for c in chunks if c.name in set(chunk_names)]
    if skip_existing:
        done = {p.name[:-len("_centroids.csv")] for p in (Path(slab_dir) / "centroids").glob("*_centroids.csv")}
        before = len(chunks)
        chunks = [c for c in chunks if c.name not in done]
        print(f"skip-existing: {before - len(chunks)} tile(s) already done, {len(chunks)} to run")
    jobs = [(c, cfg, slab_dir, deskew, adaptive, max_distance) for c in chunks]
    rows = []
    t_start = time.perf_counter()
    if workers > 1:
        print(f"running {len(jobs)} tiles on {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (msg, row) in enumerate(ex.map(_run_one, jobs, chunksize=1), 1):
                print(f"[{i}/{len(jobs)}]{msg}"); rows.append(row)
    else:
        for i, job in enumerate(jobs, 1):
            msg, row = _run_one(job)
            print(f"[{i}/{len(jobs)}]{msg}"); rows.append(row)
    if rows:
        wall = time.perf_counter() - t_start
        per = sorted(r["seconds"] for r in rows)
        print(f"TIMING: {len(rows)} tiles in {wall:.1f} s wall on {workers} worker(s) -> {wall / len(rows):.2f} s/tile "
              f"effective; single-process median {per[len(per) // 2]:.1f} s, max {per[-1]:.1f} s", flush=True)
    if rows:    # record which settings each tile used, so any output can be traced to them
        path = Path(slab_dir) / "config_choices.csv"
        new = pd.DataFrame(rows)
        if skip_existing and path.exists():
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        new.to_csv(path, index=False)
        pd.DataFrame(rows)[["seconds", "chunk"]].to_csv(Path(slab_dir) / "timing_per_chunk.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir")
    ap.add_argument("--deskew", action="store_true",
                    help="deskew each tile (-7 deg shear, see deskew.py) before segmenting. Off by default: its "
                         "effect on accuracy was inconclusive and the parameters were tuned on native tiles.")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="named parameter set from config.PRESETS, for ONE tissue type (default: the PFC-validated Stage1Config)")
    ap.add_argument("--adaptive", action="store_true",
                    help="EXPERIMENTAL: choose settings per tile from its image statistics (see docs/AUTO_CONFIG.md)")
    ap.add_argument("--max-distance", type=float, default=0.75, help="match cutoff for --adaptive; farther tiles use the defaults")
    ap.add_argument("--fast-edt", action="store_true",
                    help="opt-in: distance transform via the `edt` package (pip install edt); ~3x faster, float32")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes; match the CPUs you requested")
    ap.add_argument("--skip-existing", action="store_true", help="skip tiles that already have a centroid CSV")
    a = ap.parse_args()
    cfg = Stage1Config(**{**(PRESETS[a.preset] if a.preset else {}), **({"fast_edt": True} if a.fast_edt else {})})
    if a.preset:
        print(f"preset '{a.preset}': {PRESETS[a.preset]}", flush=True)
    run_on_slab(a.slab_dir, cfg, deskew=a.deskew, workers=a.workers, skip_existing=a.skip_existing,
                adaptive=a.adaptive, max_distance=a.max_distance)


if __name__ == "__main__":
    main()

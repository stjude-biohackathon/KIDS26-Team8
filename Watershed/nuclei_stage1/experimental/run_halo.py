"""
EXPERIMENTAL -- not part of the recommended workflow. In practice this took much longer than expected on a full
slab (see docs/VALIDATION.md), so the supported way to handle tile boundaries is `nuclei_stage1.stitch`. Kept
here, working and tested, in case a faster version is worth revisiting.

Stage 1 (threshold + watershed) with OVERLAPPING EDGES, so nuclei cut by chunk boundaries are segmented whole.

Each chunk is segmented inside a padded block = the chunk plus a margin (halo) of its neighbours' raw voxels.
A nucleus belongs to the chunk that contains its CENTRE; that chunk keeps the whole object, including the part
that sticks out into the margin. Every nucleus is therefore counted exactly once, is never cut by a chunk
boundary, and the per-slice threshold sees context across the boundary (no step in the mask at chunk edges).

Two passes (both parallel):
  1. segment   per chunk: padded block -> stage 1 -> keep objects whose centre is inside the chunk. Writes
               centroids/{chunk}_centroids.csv (ROI voxel coordinates, native frame) and small piece files
               labels_pieces/{target}__from__{chunk}.tif (the owned object voxels that fall in each chunk's
               area, ids globally unique).
  2. assemble  per chunk: paint its own piece first, then the pieces spilling in from neighbours ->
               labels_merged/{chunk}_labels.tif (same ids across tiles; marker .global_ids). Every object is
               whole, so view_slab --labels-dir labels_merged shows no lines at chunk boundaries.

Margin size is in micrometres (default 20 um = room for objects up to ~40 um long centred in the chunk; the
length cap is 60 um, anything longer is flagged anyway). In voxels that is margin_um / voxel_size per axis
(z 10, y 36, x 38 with the current config). Cost: the padded block is ~3x the voxels of a bare chunk, so expect
roughly 2-3x the stage 1 time per chunk.

Usage:
    python -m nuclei_stage1.experimental.run_halo <slab_dir> --workers 8
    python -m nuclei_stage1.experimental.run_halo <slab_dir> --workers 8 --halo-um 25 --roi 0
    python -m nuclei_stage1.experimental.run_halo <slab_dir> --workers 8 --skip-existing      # resume after a time-out
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from ..config import Stage1Config
from ..chunks import list_chunks
from ..segmentation import run_stage1

ID_STRIDE = 4096          # ids per chunk: gid = chunk_ordinal * ID_STRIDE + local id


def _box_overlap(a0, a1, b0, b1):
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    return (lo, hi) if (hi > lo).all() else None


def build_padded(c, by_off, steps, shape, halo):
    """Padded raw block around chunk c, plus a mask of real (non-padding) voxels."""
    o = np.array([c.z, c.y, c.x]); sh = np.array(shape); h = np.array(halo)
    p0, p1 = o - h, o + sh + h
    out = np.zeros(tuple(sh + 2 * h), np.float32); valid = np.zeros(out.shape, bool)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = (o[0] + dz * steps[0], o[1] + dy * steps[1], o[2] + dx * steps[2])
                nb = by_off.get(k)
                if nb is None:
                    continue
                vol = tifffile.imread(nb.raw_tif)
                n0 = np.array(k); n1 = n0 + np.array(vol.shape)
                ov = _box_overlap(p0, p1, n0, n1)
                if ov is None:
                    continue
                lo, hi = ov
                dst = tuple(slice(l - q, u - q) for l, u, q in zip(lo, hi, p0))
                src = tuple(slice(l - q, u - q) for l, u, q in zip(lo, hi, n0))
                out[dst] = vol[src]; valid[dst] = True
    return out, valid


_W = {}


def _init(by_off, steps, halo, slab, cfg):
    _W.update(by_off=by_off, steps=steps, halo=halo, slab=slab, cfg=cfg)


def segment_one(job):
    c, ordinal = job
    by_off, steps, halo, slab, cfg = (_W[k] for k in ("by_off", "steps", "halo", "slab", "cfg"))
    core = tifffile.imread(c.raw_tif).shape
    padded, valid = build_padded(c, by_off, steps, core, halo)
    labels = run_stage1(padded, cfg, valid=valid)[0]
    h = np.array(halo); sh = np.array(core)
    ids = np.unique(labels); ids = ids[ids > 0]
    rows, owned = [], np.zeros(labels.shape, np.uint32)
    if len(ids):
        cnt = np.bincount(labels.ravel())
        # centres of mass via one bincount per axis (a single pass each) instead of ndi.center_of_mass
        nlab = len(cnt); flat = labels.ravel()
        com = np.empty((len(ids), 3))
        for ax in range(3):
            idx = np.arange(labels.shape[ax], dtype=np.float64)
            shp = [1, 1, 1]; shp[ax] = -1
            w = np.broadcast_to(idx.reshape(shp), labels.shape).ravel()
            com[:, ax] = (np.bincount(flat, weights=w, minlength=nlab)[ids] / cnt[ids])
        inside = ((com >= h) & (com < h + sh)).all(axis=1)
        if inside.sum() >= ID_STRIDE:
            raise RuntimeError(f"{c.name}: {int(inside.sum())} nuclei in one chunk exceeds ID_STRIDE")
        lut = np.zeros(nlab, np.uint32)
        for j, (i, cm) in enumerate(zip(ids[inside], com[inside]), 1):
            gid = ordinal * ID_STRIDE + j
            lut[i] = gid
            rows.append({"label_id": gid, "z": cm[0] - h[0] + c.z, "y": cm[1] - h[1] + c.y,
                         "x": cm[2] - h[2] + c.x, "voxel_count": int(cnt[i])})
        owned = lut[labels]
    cd = Path(slab) / "centroids"; cd.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["label_id", "z", "y", "x", "voxel_count"]).to_csv(cd / f"{c.name}_centroids.csv", index=False)

    # split the owned labels into pieces, one per chunk area they fall in (incl. the chunk's own)
    pd_ = Path(slab) / "labels_pieces"; pd_.mkdir(exist_ok=True)
    p0 = np.array([c.z, c.y, c.x]) - h
    npieces = 0
    if rows:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    k = (c.z + dz * steps[0], c.y + dy * steps[1], c.x + dx * steps[2])
                    t = by_off.get(k)
                    if t is None:
                        continue
                    t0 = np.array(k)
                    t1 = t0 + (sh if (dz, dy, dx) == (0, 0, 0) else np.array(steps))
                    ov = _box_overlap(p0, p0 + np.array(owned.shape), t0, t1)
                    if ov is None:
                        continue
                    lo, hi = ov
                    piece = owned[tuple(slice(l - q, u - q) for l, u, q in zip(lo, hi, p0))]
                    if piece.any():
                        full = np.zeros(tuple(t1 - t0), np.uint32)
                        full[tuple(slice(l - q, u - q) for l, u, q in zip(lo, hi, t0))] = piece
                        tifffile.imwrite(pd_ / f"{t.name}__from__{c.name}.tif", full, compression="zlib")
                        npieces += 1
    return c.name, len(rows), npieces


def assemble_one(c):
    slab = _W["slab"]
    core = tifffile.imread(c.raw_tif).shape
    out = np.zeros(core, np.uint32)
    pieces = sorted((Path(slab) / "labels_pieces").glob(f"{c.name}__from__*.tif"),
                    key=lambda p: (p.name.split("__from__")[1][:-4] != c.name, p.name))   # own piece first
    for p in pieces:
        t = tifffile.imread(p)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, core))
        t = t[sl]; dst = out[sl]
        dst[(dst == 0) & (t > 0)] = t[(dst == 0) & (t > 0)]
    od = Path(slab) / "labels_merged"; od.mkdir(exist_ok=True)
    tifffile.imwrite(od / f"{c.name}_labels.tif", out, compression="zlib")
    return c.name, int(len(np.unique(out)) - (1 if (out == 0).any() else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir")
    ap.add_argument("--roi", default=None)
    ap.add_argument("--halo-um", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    cfg = Stage1Config(); vs = np.array(cfg.voxel_size, float)
    halo = tuple(int(np.ceil(a.halo_um / v)) for v in vs)
    print(f"halo {a.halo_um:g} um -> {halo} voxels (z,y,x)")
    slab = Path(a.slab_dir)
    groups = {}
    for c in list_chunks(slab):
        groups.setdefault(c.roi_stem, []).append(c)
    items = sorted(groups.items())
    if a.roi is not None:
        items = [g for i, g in enumerate(items) if str(i) == a.roi] or [g for g in items if a.roi in g[0]]
    if not items:
        sys.exit("no matching chunks")

    for sub in ("labels_pieces", "labels_merged"):          # stale pieces from an earlier run would pollute assembly
        if not a.skip_existing and (slab / sub).exists():
            for f in (slab / sub).glob("*.tif"):
                f.unlink()
    ordinal0 = 0
    for roi, cs in items:
        cs = sorted(cs, key=lambda c: (c.z, c.y, c.x))
        by_off = {(c.z, c.y, c.x): c for c in cs}
        ordinal = {c.name: ordinal0 + i for i, c in enumerate(cs)}
        steps = []
        for ax, key in enumerate(("z", "y", "x")):
            u = sorted({getattr(c, key) for c in cs})
            d = [b - a_ for a_, b in zip(u, u[1:])]
            steps.append(min(d) if d else int(tifffile.imread(cs[0].raw_tif).shape[ax]))
        steps = tuple(int(x) for x in steps)
        if any(h > st for h, st in zip(halo, steps)):
            sys.exit(f"halo {halo} exceeds the tile step {steps}; use a smaller --halo-um")
        print(f"ROI {roi}: {len(cs)} chunks, tile step {steps}")
        todo = cs
        if a.skip_existing:
            todo = [c for c in cs if not (slab / "centroids" / f"{c.name}_centroids.csv").exists()]
            print(f"  skip-existing: {len(cs) - len(todo)} done, {len(todo)} to segment")
        if a.workers > 1:
            ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(by_off, steps, halo, slab, cfg))
            mapper = lambda f, js: ex.map(f, js, chunksize=1)          # noqa: E731
        else:
            _init(by_off, steps, halo, slab, cfg)
            mapper = lambda f, js: map(f, js)                          # noqa: E731
        for i, (name, n, npieces) in enumerate(mapper(segment_one, [(c, ordinal[c.name]) for c in todo]), 1):
            print(f"  [{i}/{len(todo)}] {name}: {n} nuclei owned ({npieces} piece files)")
        print("  segmented; assembling whole-object tiles ...")
        for i, (name, n) in enumerate(mapper(assemble_one, cs), 1):
            if i % 100 == 0 or i == len(cs):
                print(f"  assembled {i}/{len(cs)}")
        if a.workers > 1:
            ex.shutdown()
        ordinal0 += len(cs)
    (slab / "labels_merged").mkdir(exist_ok=True)
    (slab / "labels_merged" / ".global_ids").write_text("globally unique ids; objects are whole across chunk boundaries\n")
    print(f"\ndone: {slab / 'labels_merged'}  (view with: python -m nuclei_stage1.view_slab <slab> --roi 0 --labels-dir labels_merged)")


if __name__ == "__main__":
    main()

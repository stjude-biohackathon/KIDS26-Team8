"""
Stitch per-chunk labels into whole-slab objects.

Chunks are segmented independently, so a nucleus cut by a chunk boundary comes out as 2 (or 4, or 8 at a
corner) separate labels with different colours / ids / centroids. This merges them back:

  For every pair of face-adjacent chunks, compare the LAST plane of one with the FIRST plane of the other.
  Label a (chunk A) and label b (chunk B) are the same nucleus if their footprints on those planes overlap:
      overlap pixels >= --min-overlap   AND   overlap / min(footprint_a, footprint_b) >= --min-containment
  (a nucleus cut by a plane has near-identical cross-sections on both sides; two different nuclei that merely
  touch across the plane overlap only a little). Pairs are merged strongest first with union-find, and a merge
  is REFUSED if the combined object would be longer than --max-extent-um (default: config.max_extent_um, else
  60), so a chain of touching nuclei can't snowball into one giant object.

Reads   <slab>/<labels-dir>/{chunk}_labels.tif           (default labels/)
Writes  <slab>/<out-dir>/{chunk}_labels.tif              (default labels_stitched/; ids are GLOBAL and unique,
                                                         marker file .global_ids tells view_slab)
        <slab>/<out-dir>/stitched_centroids.csv          one row per stitched nucleus (ROI voxel coordinates,
                                                         native frame, same convention as centroids/)

Limits: pieces that stage 1 already DROPPED (a sliver under min_object_voxels, or below the "too small" volume
floor) cannot be recovered by stitching. If that matters, see nuclei_stage1/experimental/run_halo.py.

Usage:
    python -m nuclei_stage1.stitch <slab_dir> --workers 8
    python -m nuclei_stage1.stitch <slab_dir> --roi 0 --min-containment 0.4
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from .config import Stage1Config
from .chunks import list_chunks


def scan_chunk(path):
    lab = tifffile.imread(path)
    faces = {}
    for ax in range(3):
        faces[(ax, 0)] = np.take(lab, 0, axis=ax).astype(np.int64)
        faces[(ax, 1)] = np.take(lab, lab.shape[ax] - 1, axis=ax).astype(np.int64)
    stats = {}
    if lab.max() > 0:
        ids = np.unique(lab)
        ids = ids[ids > 0]
        objs = ndi.find_objects(lab)
        com = ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, ids)
        cnt = np.bincount(lab.ravel())
        for i, c in zip(ids, com):
            s = objs[i - 1]
            stats[int(i)] = (int(cnt[i]), *map(float, c),
                             *(sl.start for sl in s), *(sl.stop for sl in s))
    return lab.shape, faces, stats


class UF:
    def __init__(self):
        self.p, self.cnt, self.sum, self.lo, self.hi = [], [], [], [], []

    def add(self, count, c, lo, hi):
        self.p.append(len(self.p)); self.cnt.append(count)
        self.sum.append(np.array(c) * count); self.lo.append(np.array(lo, float)); self.hi.append(np.array(hi, float))
        return len(self.p) - 1

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def merged_extent(self, a, b, vs):
        lo = np.minimum(self.lo[a], self.lo[b]); hi = np.maximum(self.hi[a], self.hi[b])
        return float(((hi - lo) * vs).max())

    def union(self, a, b):
        self.p[b] = a; self.cnt[a] += self.cnt[b]; self.sum[a] = self.sum[a] + self.sum[b]
        self.lo[a] = np.minimum(self.lo[a], self.lo[b]); self.hi[a] = np.maximum(self.hi[a], self.hi[b])


def face_edges(fa, fb, min_overlap, min_containment):
    h, w = min(fa.shape[0], fb.shape[0]), min(fa.shape[1], fb.shape[1])
    fa, fb = fa[:h, :w], fb[:h, :w]
    both = (fa > 0) & (fb > 0)
    if not both.any():
        return []
    a, b = fa[both], fb[both]
    key = a * (1 << 31) + b
    uk, n = np.unique(key, return_counts=True)
    la, lb = uk // (1 << 31), uk % (1 << 31)
    size_a = dict(zip(*np.unique(fa[fa > 0], return_counts=True)))
    size_b = dict(zip(*np.unique(fb[fb > 0], return_counts=True)))
    out = []
    for x, y, c in zip(la, lb, n):
        cont = c / min(size_a[x], size_b[y])
        if c >= min_overlap and cont >= min_containment:
            out.append((float(cont), int(x), int(y), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--out-dir", default="labels_stitched")
    ap.add_argument("--roi", default=None, help="ROI index or part of its name (default: all ROIs)")
    ap.add_argument("--min-overlap", type=int, default=10)
    ap.add_argument("--min-containment", type=float, default=0.5)
    ap.add_argument("--max-extent-um", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    cfg = Stage1Config()
    vs = np.array(cfg.voxel_size, float)
    cap = a.max_extent_um if a.max_extent_um is not None else (getattr(cfg, "max_extent_um", 0.0) or 60.0)
    slab, ld, od = Path(a.slab_dir), a.labels_dir, a.out_dir

    groups = {}
    for c in list_chunks(slab):
        if (slab / ld / f"{c.name}_labels.tif").exists():
            groups.setdefault(c.roi_stem, []).append(c)
    if not groups:
        sys.exit(f"no {ld}/*_labels.tif found in {slab}")
    items = sorted(groups.items())
    if a.roi is not None:
        items = [g for i, g in enumerate(items) if str(i) == a.roi] or [g for g in items if a.roi in g[0]]
    (slab / od).mkdir(exist_ok=True)
    (slab / od / ".global_ids").write_text("labels in this folder have globally unique ids (stitched)\n")

    next_gid, all_rows, tot_before, tot_after = 1, [], 0, 0
    for roi, cs in items:
        print(f"ROI {roi}: {len(cs)} chunks")
        paths = [slab / ld / f"{c.name}_labels.tif" for c in cs]
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                scans = list(ex.map(scan_chunk, paths, chunksize=4))
        else:
            scans = [scan_chunk(p) for p in paths]

        uf, node, by_chunk = UF(), {}, {}
        index = {(c.z, c.y, c.x): i for i, c in enumerate(cs)}
        for i, (c, (shape, faces, stats)) in enumerate(zip(cs, scans)):
            off = np.array([c.z, c.y, c.x], float)
            for lab_id, (n, cz, cy, cx, z0, y0, x0, z1, y1, x1) in stats.items():
                node[(i, lab_id)] = by_chunk.setdefault(i, {})[lab_id] = uf.add(
                    n, off + [cz, cy, cx], off + [z0, y0, x0], off + [z1, y1, x1])
        before = len(node)

        edges = []
        for i, (c, (shape, faces, _)) in enumerate(zip(cs, scans)):
            for ax in range(3):
                nb_off = [c.z, c.y, c.x]; nb_off[ax] += shape[ax]
                j = index.get(tuple(nb_off))
                if j is None:
                    continue
                for cont, la, lb, n in face_edges(faces[(ax, 1)], scans[j][1][(ax, 0)], a.min_overlap, a.min_containment):
                    edges.append((cont, n, node[(i, la)], node[(j, lb)]))
        edges.sort(key=lambda e: (-e[0], -e[1]))
        merged = refused = 0
        for cont, n, u, v in edges:
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue
            if cap > 0 and uf.merged_extent(ru, rv, vs) > cap:
                refused += 1
                continue
            uf.union(ru, rv); merged += 1
        roots = {}
        for k in node:
            roots.setdefault(uf.find(node[k]), None)
        gid = {r: next_gid + i for i, r in enumerate(roots)}
        after = len(roots)
        print(f"  {before} chunk-level labels -> {after} nuclei  ({merged} merges, {refused} refused by the "
              f"{cap:g} um length cap, {len(edges)} candidate pairs)")
        tot_before += before; tot_after += after

        for i, (c, p) in enumerate(zip(cs, paths)):
            lab = tifffile.imread(p)
            lut = np.zeros(int(lab.max()) + 1, np.uint32)
            for l, nd in by_chunk.get(i, {}).items():
                lut[l] = gid[uf.find(nd)]
            tifffile.imwrite(slab / od / f"{c.name}_labels.tif", lut[lab], compression="zlib")
        for r, g in gid.items():
            cx = uf.sum[r] / uf.cnt[r]; ext = (uf.hi[r] - uf.lo[r]) * vs
            all_rows.append({"label_id": g, "roi": roi, "z": cx[0], "y": cx[1], "x": cx[2],
                             "voxel_count": uf.cnt[r], "longest_extent_um": float(ext.max())})
        next_gid += after

    pd.DataFrame(all_rows).to_csv(slab / od / "stitched_centroids.csv", index=False)
    print(f"\ntotal: {tot_before} -> {tot_after} nuclei; wrote {slab / od}")


if __name__ == "__main__":
    main()

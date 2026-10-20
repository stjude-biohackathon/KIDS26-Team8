"""
Are the voxel sizes plausible? Measures the nuclei in a label volume and asks what voxel size would make them round.

Real nuclei are roughly round, so the median bounding-box extent of the segmented nuclei, MULTIPLIED by the voxel size
on each axis, should be about the same on all three axes. If it is not, either the voxel size is wrong or the
segmentation is systematically stretched along one axis. This does not give absolute sizes (a nucleus's true diameter is
not known from the labels alone); it gives the RATIO between axes, which is what makes nuclei look flat or round.

Works on any label volume(s): a corrected ground-truth chunk, or tiles from labels/ or labels_merged/. Nuclei touching a
tile edge are skipped (they are cut). Camera shear widens x slightly (about 0.12 plane per z plane) -- reported but
not corrected.

Usage:
    python -m nuclei_stage1.check_voxel_scale gt_labels.tif
    python -m nuclei_stage1.check_voxel_scale /path/to/slab/labels_merged --n-tiles 40
    python -m nuclei_stage1.check_voxel_scale labels.tif --voxel-size 2.0,0.5616,0.84      # judge a candidate instead of config
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .config import Stage1Config


def extents(path):
    lab = tifffile.imread(path)
    out = []
    if lab.max() == 0:
        return np.zeros((0, 3)), 0
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        touches = any(s.start == 0 or s.stop == n for s, n in zip(sl, lab.shape))
        if not touches:
            out.append([s.stop - s.start for s in sl])
    return np.array(out, float).reshape(-1, 3), int(len(np.unique(lab)) - (1 if (lab == 0).any() else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a label .tif, or a folder of *_labels.tif tiles")
    ap.add_argument("--n-tiles", type=int, default=30, help="max tiles to read from a folder (evenly spaced)")
    ap.add_argument("--voxel-size", default=None, help="z,y,x um to test (default config.voxel_size)")
    ap.add_argument("--min-voxels", type=int, default=0, help="ignore objects with a smaller bbox volume")
    a = ap.parse_args()

    vs = np.array([float(t) for t in a.voxel_size.split(",")], float) if a.voxel_size else np.array(Stage1Config().voxel_size, float)
    p = Path(a.path)
    files = [p] if p.is_file() else sorted(p.glob("*_labels.tif"))
    if not files:
        sys.exit(f"no label tifs found at {p}")
    if len(files) > a.n_tiles:
        files = [files[i] for i in np.linspace(0, len(files) - 1, a.n_tiles).astype(int)]
    ext, total = [], 0
    for f in files:
        e, n = extents(f)
        ext.append(e); total += n
    ext = np.concatenate(ext) if ext else np.zeros((0, 3))
    if len(ext) < 20:
        sys.exit(f"only {len(ext)} uncut nuclei measured from {len(files)} file(s) -- need more tiles")
    ext = ext[np.prod(ext, axis=1) >= a.min_voxels]
    med = np.median(ext, axis=0)
    phys = med * vs
    print(f"{len(ext)} uncut nuclei measured (of {total} labels in {len(files)} file(s)); voxel size tested (z,y,x) um = {tuple(float(v) for v in vs)}")
    print(f"  median extent, voxels   z {med[0]:.1f}   y {med[1]:.1f}   x {med[2]:.1f}")
    print(f"  median extent, um       z {phys[0]:.1f}   y {phys[1]:.1f}   x {phys[2]:.1f}")
    ratio = phys / phys[1]
    print(f"  relative to y           z {ratio[0]:.2f}   y 1.00   x {ratio[2]:.2f}     (round nuclei -> all about 1.00)")
    implied = med[1] * vs[1] / med                      # voxel size that equalises all three, anchored on y
    print(f"  voxel size that would make them round (anchored on y = {vs[1]:g}): "
          f"({implied[0]:.3g}, {implied[1]:.3g}, {implied[2]:.3g}) um")
    bad = [n for n, r in zip("zx", (ratio[0], ratio[2])) if abs(r - 1) > 0.25]
    if bad:
        print(f"  -> the {'/'.join(bad)} axis is more than 25% off round: nuclei will look {'squashed' if min(ratio) < 1 else 'stretched'} "
              f"along it with this voxel size. Check the acquisition metadata, or view with --voxel-size.")
    else:
        print("  -> within 25% of round on every axis.")


if __name__ == "__main__":
    main()

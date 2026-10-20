"""Synthetic volumes shared by the tests."""
import numpy as np
import tifffile


def add_ellipsoid(vol, centre, radii, value):
    zz, yy, xx = np.indices(vol.shape)
    d = ((zz - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 + ((xx - centre[2]) / radii[2]) ** 2
    inside = d <= 1
    # a little texture: the per-slice percentile threshold is strict (>), so a perfectly flat plateau would tie with it
    vol[inside] = value + np.random.default_rng(int(sum(centre))).normal(0, 30.0, inside.sum())
    return vol


def noisy(shape, seed=0, level=100.0, sd=10.0):
    return np.random.default_rng(seed).normal(level, sd, shape).astype(np.float32)


def make_slab(root, nuclei, tiles=((0, 0, 0), (0, 128, 0)), shape=(128, 128, 128), seed=0):
    """A fake slab with raw_chunked/ tiles of one ROI. `nuclei` = list of (centre_zyx_in_ROI_coords, radii_vox).
    Every tile sees the same ROI-wide field, so a nucleus on a boundary is cut identically in both tiles."""
    ext = np.array([max(t[i] for t in tiles) + shape[i] for i in range(3)])
    field = noisy(tuple(ext), seed)
    for c, r in nuclei:
        add_ellipsoid(field, c, r, 1000.0)
    (root / "raw_chunked").mkdir(parents=True)
    for t in tiles:
        sl = tuple(slice(t[i], t[i] + shape[i]) for i in range(3))
        tifffile.imwrite(root / "raw_chunked" / f"roiA_z{t[0]:06d}_y{t[1]:06d}_x{t[2]:06d}.tif", field[sl].astype(np.uint16))
    return root

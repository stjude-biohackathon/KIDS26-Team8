"""
Deskew a native nuclei-camera chunk (or its labels), and map points between
native and deskewed coordinates.

Geometry (from the registration transforms file, nuclei camera = channel 0):
the deskew is a pure shear of the scan axis X against camera axis Z, angle
-7 deg => |slope| = tan(7 deg) = 0.1228 index-units of X per Z pixel.

DIRECTION WAS MEASURED, NOT ASSUMED. Applying the printed 4x4 as a forward
map (X' = X - 0.1228 Z) DOUBLED the shear present in the annotated nuclei
(residual X-vs-Z slope -0.12 -> -0.23). Applying its inverse
(X' = X + 0.1228 Z) removed it (residual +0.02 +/- 0.06, median 0.003).
So the printed matrix describes the shear the native data carries, and
deskewing means undoing it. Based on one annotated PFC chunk (113 nuclei);
re-run `python deskew.py --check labels.tif` on any new GT chunk to confirm.

What this does NOT do: the rest of the channel-0 registration (scale 0.49,
rotation, shear, translation onto the splitter-camera grid). That part's
direction is not validated by this check and needs the other channels to
verify alignment; it is also unnecessary for segmenting nuclei at native
resolution. Map centroids into the registered frame afterwards instead.

Output is padded in X (~16 planes for a 128-plane chunk). The padding is
returned as a `valid` mask: pass it to stage 1 (run_stage1(..., valid=valid))
so the zero wedge is not counted in the per-slice percentile threshold.

Usage:
    python deskew.py raw_chunk.tif deskewed_raw.tif            # raw intensities (linear interp)
    python deskew.py labels.tif deskewed_labels.tif --labels   # label ids (nearest neighbour)
    python deskew.py --check labels.tif                        # residual shear before/after
"""
import argparse

import numpy as np
import tifffile
from scipy import ndimage as ndi

# nuclei camera deskew angle, degrees (splitter cameras use -14; not relevant here)
NUCLEI_DESKEW_DEG = -7.0
# X' = X + SLOPE * Z  (direction validated empirically -- see module docstring)
SLOPE = abs(np.tan(np.deg2rad(NUCLEI_DESKEW_DEG)))


def _geometry(shape, slope=SLOPE):
    """Output X extent and X translation so every shifted plane lands at X' >= 0."""
    z = shape[0]
    shifts = np.array([0.0, slope * (z - 1)])
    tx = -min(0.0, shifts.min())
    new_x = int(np.ceil(shape[2] - 1 + shifts.max() + tx)) + 1
    return tx, new_x


def deskew_volume(vol, is_labels=False, slope=SLOPE, nearest=False):
    """Returns (deskewed, valid). `valid` is True where a voxel is real data.
    nearest=True shifts whole planes by rounded integer amounts (no resampling blur)
    instead of linear interpolation -- labels always use it. Measured on the annotated
    PFC chunk, neither is clearly better for stage-1 F1 (see module docstring)."""
    tx, new_x = _geometry(vol.shape, slope)
    L = np.array([[1.0, 0, 0], [0, 1, 0], [slope, 0, 1]])      # native -> deskewed, (Z,Y,X)
    Linv = np.linalg.inv(L)
    t = np.array([0.0, 0.0, tx])
    kw = dict(offset=-Linv @ t, output_shape=(vol.shape[0], vol.shape[1], new_x),
              mode="constant", cval=0)
    if is_labels or nearest:
        out = ndi.affine_transform(vol, Linv, order=0, **kw)
    else:
        out = ndi.affine_transform(vol.astype(np.float32), Linv, order=1, **kw)
        if np.issubdtype(vol.dtype, np.integer):
            info = np.iinfo(vol.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        out = out.astype(vol.dtype)
    valid = ndi.affine_transform(np.ones(vol.shape, np.float32), Linv, order=1, **kw) > 0.999
    return out, valid


def to_deskewed_points(zyx, shape, slope=SLOPE):
    """Native (z, y, x) point(s) -> deskewed coordinates (same shape convention
    as the volume returned by deskew_volume for a volume of `shape`)."""
    p = np.atleast_2d(np.asarray(zyx, float)).copy()
    tx, _ = _geometry(shape, slope)
    p[:, 2] = p[:, 2] + slope * p[:, 0] + tx
    return p


def to_native_points(zyx, shape, slope=SLOPE):
    """Inverse of to_deskewed_points: map centroids found in a deskewed volume
    back to native chunk coordinates (needed before adding the chunk's global
    offset for slab-global centroids)."""
    p = np.atleast_2d(np.asarray(zyx, float)).copy()
    tx, _ = _geometry(shape, slope)
    p[:, 2] = p[:, 2] - slope * p[:, 0] - tx
    return p


def residual_shear(labels, min_voxels=200):
    """Mean X-vs-Z regression slope (index units) of each instance's voxels.
    ~0 for an upright population; ~ -0.12 in native frames, per the check above."""
    z, _, x = np.indices(labels.shape)
    slopes = []
    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        if m.sum() < min_voxels:
            continue
        Z, X = z[m].astype(float), x[m].astype(float)
        slopes.append(np.mean((Z - Z.mean()) * (X - X.mean())) / np.var(Z))
    s = np.array(slopes)
    return len(s), float(s.mean()), float(np.median(s)), float(s.std() / np.sqrt(len(s)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--labels", action="store_true", help="input is a label volume (nearest-neighbour)")
    ap.add_argument("--nearest", action="store_true",
                    help="raw input: shift whole planes by rounded integers instead of linear interpolation")
    ap.add_argument("--check", action="store_true",
                    help="report residual shear of a label volume before and after deskewing")
    a = ap.parse_args()

    vol = tifffile.imread(a.input)
    if a.check:
        n, m, med, sem = residual_shear(vol)
        n2, m2, med2, sem2 = residual_shear(deskew_volume(vol, True)[0])
        print(f"native:    n={n} mean slope={m:+.4f} median={med:+.4f} sem={sem:.4f}")
        print(f"deskewed:  n={n2} mean slope={m2:+.4f} median={med2:+.4f} sem={sem2:.4f}")
        print("deskew is helping if the deskewed slope is near 0 (within ~2 sem) and smaller in "
              "magnitude than native.")
    else:
        if not a.output:
            ap.error("output path required unless --check")
        out, valid = deskew_volume(vol, a.labels, nearest=a.nearest)
        tifffile.imwrite(a.output, out)
        print(f"{a.input} {vol.shape} -> {a.output} {out.shape}; "
              f"{(~valid).mean():.1%} of output voxels are padding")


if __name__ == "__main__":
    main()

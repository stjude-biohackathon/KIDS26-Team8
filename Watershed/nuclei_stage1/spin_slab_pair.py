"""
Save a SPINNING 3-D movie of a whole ROI: the raw volume on the left, the same volume with the nuclei filled in
(one colour per nucleus) on the right, rotating together. Like view_chunk_3d.py, but written to a GIF (and MP4 if
imageio-ffmpeg is installed) so it needs no display and no napari -- it runs on a cluster node.

How it is drawn (pure numpy, orthographic camera): the downsampled ROI mosaic (same builder as view_slab) is
turned into 3-D points in micrometres using config.voxel_size (or --voxel-size z,y,x), rotated about the vertical axis, tilted by a fixed
elevation, and splatted onto the image. Left panel = maximum-intensity projection of the brightest raw voxels.
Right panel = a dimmed copy of that raw image with every labelled voxel drawn on top in its nucleus colour, nearer
voxels brighter and hiding farther ones (a depth buffer), so you see the nuclei as solid coloured volumes.
It is a point-splat render, not napari's isosurface shading, so it looks flatter but rotates the same data.

Reads  <slab>/raw_chunked/*.tif  and  <slab>/<labels-dir>/{chunk}_labels.tif   (as view_slab; respects the
.global_ids marker). Writes <out>/<slab>_<roi>_spin.gif (+ .mp4 if possible) and a still _spin_poster.png.

Usage:
    python -m nuclei_stage1.spin_slab_pair <slab_dir>                                   # lists ROIs, exits
    python -m nuclei_stage1.spin_slab_pair <slab_dir> --roi all --labels-dir labels_stitched
    python -m nuclei_stage1.spin_slab_pair <slab_dir> --roi 0 --downsample 4 --turn-seconds 30 --size 700 --elevation 25
    python -m nuclei_stage1.spin_slab_pair <slab_dir> --roi 0 --no-deskew

If a ROI is too big, raise --downsample (6-8 is fine for an overview). Small nuclei thin out at large strides.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import Stage1Config
from .deskew import SLOPE
from .chunks import list_chunks
from .view_slab import assemble, extent, parse_voxel_size, roi_groups

MAX_RAW_POINTS = 6_000_000
MAX_LABEL_POINTS = 8_000_000


def label_colours(ids):
    """Distinct uint8 RGB per label id (golden-ratio hue)."""
    from matplotlib.colors import hsv_to_rgb
    ids = ids.astype(np.int64)
    h = (ids * 0.61803398875) % 1.0
    s = 0.55 + 0.4 * ((ids * 7) % 5) / 4.0
    v = 0.80 + 0.2 * ((ids * 3) % 4) / 3.0
    return (hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255).astype(np.uint8)


def build_points(raw, lab, pix_um, raw_pct, rng, shear=0.0, clip_um=0.0):
    """Return centred micrometre coordinates + values for the bright raw voxels and the labelled voxels.
    shear: deskew in mosaic index units (x += shear * z), applied to the points -- same map as view_slab --deskew,
    but without resampling the volume."""
    nz = raw[raw > 0]
    if nz.size == 0:
        raise SystemExit("raw mosaic is empty")
    thr = np.percentile(nz[:: max(1, nz.size // 2_000_000)], raw_pct)
    rz = np.nonzero(raw > thr)
    if rz[0].size > MAX_RAW_POINTS:
        keep = rng.choice(rz[0].size, MAX_RAW_POINTS, replace=False)
        rz = tuple(a[keep] for a in rz)
    rv = raw[rz].astype(np.float32)
    lo, hi = np.percentile(rv, [1, 99.8])
    rv = (np.clip((rv - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    order = np.argsort(rv, kind="stable")                       # ascending: last write = brightest = the MIP
    ridx = np.stack(rz, axis=1)[order].astype(np.float64)
    rv = rv[order]

    lz = np.nonzero(lab)
    if lz[0].size > MAX_LABEL_POINTS:
        keep = rng.choice(lz[0].size, MAX_LABEL_POINTS, replace=False)
        lz = tuple(a[keep] for a in lz)
    lids = lab[lz]
    lidx = np.stack(lz, axis=1).astype(np.float64)
    lcol = label_colours(lids)

    for idx in (ridx, lidx):
        idx[:, 2] += shear * idx[:, 0]                          # deskew (display): x' = x + slope * z
    both = np.concatenate([ridx, lidx]) * pix_um                # micrometres, physical voxel size
    centre = (both.min(axis=0) + both.max(axis=0)) / 2.0
    rpts = (ridx * pix_um - centre).astype(np.float32)
    lpts = (lidx * pix_um - centre).astype(np.float32)
    if clip_um > 0:                                              # a plate through the middle of the coarse axis
        kr, kl = np.abs(rpts[:, 0]) <= clip_um / 2, np.abs(lpts[:, 0]) <= clip_um / 2
        rpts, rv, lpts, lcol = rpts[kr], rv[kr], lpts[kl], lcol[kl]
        if not len(lpts):
            raise SystemExit("--clip-um removed every nucleus voxel; use a larger value")
        lids = lids[kl]
    return rpts, rv, lpts, lcol, int(len(np.unique(lids)))


def project(pts, theta, phi, scale, size):
    """Orthographic camera: rotate about the vertical (y) axis by theta, tilt by phi, centre on the panel.
    pts columns are (z, y, x) micrometres. Returns pixel x, pixel y, depth (smaller = nearer)."""
    z, y, x = pts[:, 0], pts[:, 1], pts[:, 2]
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    X = x * ct + z * st
    D = -x * st + z * ct
    Y = y * cp - D * sp
    D2 = y * sp + D * cp
    return (X * scale + size / 2).astype(np.int32), (Y * scale + size / 2).astype(np.int32), D2


def splat(img, px, py, vals, size, k):
    for dy in range(k):
        for dx in range(k):
            xx, yy = px + dx, py + dy
            ok = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
            img[yy[ok], xx[ok]] = vals[ok]


def render_frame(theta, phi, scale, size, k, rpts, rv, lpts, lcol):
    # left: raw MIP (points are pre-sorted by intensity, so the last write at a pixel is the brightest)
    px, py, _ = project(rpts, theta, phi, scale, size)
    raw_img = np.zeros((size, size), np.uint8)
    splat(raw_img, px, py, rv, size, k)
    right = np.repeat((raw_img * 0.30).astype(np.uint8)[..., None], 3, axis=2)
    # right: nuclei with a depth buffer: draw far -> near so the nearest voxel wins; nearer = brighter
    if len(lpts):
        qx, qy, d = project(lpts, theta, phi, scale, size)
        dmin, dmax = d.min(), d.max()
        dq = ((d - dmin) / max(dmax - dmin, 1e-6) * 65535).astype(np.uint16)
        order = np.argsort(np.uint16(65535) - dq, kind="stable")      # far first
        shade = (1.0 - 0.55 * (dq[order] / 65535.0)).astype(np.float32)
        cols = (lcol[order] * shade[:, None]).astype(np.uint8)
        splat(right, qx[order], qy[order], cols, size, k)
    return np.repeat(raw_img[..., None], 3, axis=2), right


def nice_bar(um_per_px, size):
    for um in (2000, 1000, 500, 200, 100, 50):
        if um / um_per_px <= 0.35 * size:
            return um, int(um / um_per_px)
    return None, 0


def compose(left, right, titles, bar):
    from PIL import Image, ImageDraw
    size = left.shape[0]
    band = 26
    W, H = -(-(size * 2 + 6) // 16) * 16, -(-(size + band) // 16) * 16      # multiples of 16 so MP4 encoders do not resize
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(Image.fromarray(left), (0, band)); canvas.paste(Image.fromarray(right), (size + 6, band))
    d = ImageDraw.Draw(canvas)
    d.text((6, 7), titles[0], fill=(255, 255, 255)); d.text((size + 12, 7), titles[1], fill=(255, 255, 255))
    um, px = bar
    if um:
        y = band + size - 14
        d.line([(size + 6 + size - px - 12, y), (size + 6 + size - 12, y)], fill=(255, 255, 255), width=3)
        d.text((size + 6 + size - px - 12, y - 14), f"{um / 1000:g} mm" if um >= 1000 else f"{um} um", fill=(255, 255, 255))
    return canvas


def write_movie(frames, path_base, fps):
    from PIL import Image
    gif = Path(f"{path_base}.gif")
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=False)
    out = [gif]
    try:
        import imageio.v2 as iio
        with iio.get_writer(f"{path_base}.mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(np.asarray(f))
        out.append(Path(f"{path_base}.mp4"))
    except Exception as e:                                           # noqa: BLE001 -- mp4 is a bonus
        print(f"   (no mp4: {type(e).__name__}; GIF written)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir")
    ap.add_argument("--roi", default=None, help="ROI index, part of its name, or 'all'")
    ap.add_argument("--voxel-size", default=None, help="voxel size in um as z,y,x for display (default: config.voxel_size). Use to test a different value, e.g. 2.0,0.5616,0.84")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--downsample", type=int, default=4, help="stride in y and x (default 4)")
    ap.add_argument("--z-downsample", type=int, default=1)
    ap.add_argument("--turn-seconds", type=float, default=20.0, help="seconds for one full rotation (default 20)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frames", type=int, default=None, help="frames in one turn (default: fps x turn-seconds)")
    ap.add_argument("--size", type=int, default=640, help="each panel is size x size pixels")
    ap.add_argument("--elevation", type=float, default=25.0, help="fixed camera tilt, degrees")
    ap.add_argument("--raw-percentile", type=float, default=97.0,
                    help="raw voxels above this percentile are drawn in the raw panel (default 97)")
    ap.add_argument("--out", default=None, help="output folder (default <slab>/figures)")
    ap.add_argument("--clip-um", type=float, default=0.0,
                    help="draw only a plate this many um thick through the middle of the coarse axis (both panels). "
                         "A whole ROI is millimetres deep, so a full-depth view is solid with nuclei; a 100-300 um plate "
                         "shows individual nuclei and the empty space around them. 0 = whole depth (default)")
    ap.add_argument("--no-deskew", action="store_true",
                    help="show the native (sheared) frame. Default is the corrected view: physical voxel size AND the "
                         "camera shear undone (display only), like view_slab --deskew")
    ap.add_argument("--max-gb", type=float, default=6.0)
    a = ap.parse_args()

    slab = Path(a.slab_dir)
    groups = roi_groups(list_chunks(slab))
    if not groups:
        sys.exit(f"no chunks found in {slab}/raw_chunked")
    if a.roi is None:
        print(f"{len(groups)} ROI(s):")
        for i, (stem, cs) in enumerate(groups):
            Z, Y, X = extent(cs)
            print(f"  [{i}] {stem}: {len(cs)} chunks, extent z{Z} y{Y} x{X} voxels")
        sys.exit("\nre-run with --roi <index> or --roi all")
    sel = list(enumerate(groups)) if a.roi == "all" else (
        [(i, g) for i, g in enumerate(groups) if str(i) == a.roi] or [(i, g) for i, g in enumerate(groups) if a.roi in g[0]])[:1]
    if not sel:
        sys.exit(f"no ROI matches {a.roi!r}")

    vs = parse_voxel_size(a.voxel_size, Stage1Config().voxel_size)
    dz, dxy = a.z_downsample, a.downsample
    pix_um = vs * np.array([dz, dxy, dxy])
    out_dir = Path(a.out) if a.out else slab / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for idx, (stem, cs) in sel:
        Z, Y, X = extent(cs)
        gb = (-(-Z // dz)) * (-(-Y // dxy)) * (-(-X // dxy)) * 6 / 1e9
        print(f"[{idx}] {stem}: {len(cs)} chunks, mosaic ~{gb:.1f} GB")
        if gb > a.max_gb:
            print(f"   skipped: > {a.max_gb} GB. Raise --downsample (or --max-gb).")
            continue
        t0 = time.time()
        raw, lab = assemble(cs, slab, a.labels_dir, dz, dxy)
        shear = 0.0 if a.no_deskew else SLOPE * dz / dxy
        rpts, rv, lpts, lcol, n_vis = build_points(raw, lab, pix_um, a.raw_percentile, rng, shear, a.clip_um)
        del raw, lab
        # fit the bounding sphere of the volume in the panel
        ext = np.abs(np.concatenate([rpts, lpts])).max(axis=0)
        radius = float(np.sqrt((ext ** 2).sum()))
        scale = 0.47 * a.size / radius                                   # pixels per micrometre
        um_per_px = 1.0 / scale
        k = max(1, int(np.ceil(pix_um.max() * scale)))                   # splat width so voxels leave no holes
        bar = nice_bar(um_per_px, a.size)
        tag = ("native frame" if a.no_deskew else "deskewed") + f", voxel {tuple(round(float(v), 3) for v in vs)}" + (f", {a.clip_um:g} um plate" if a.clip_um else "")
        titles = (f"{stem}  raw  [{tag}]", f"nuclei ({n_vis} visible, from {a.labels_dir}/)")
        n_frames = a.frames or max(8, int(round(a.fps * a.turn_seconds)))
        print(f"   {len(rpts):,} raw points, {len(lpts):,} nucleus voxels, {n_vis} nuclei; rendering {n_frames} frames ({n_frames / a.fps:.0f} s per turn)")
        frames = []
        for i in range(n_frames):
            th = 2 * np.pi * i / n_frames
            l_img, r_img = render_frame(th, np.deg2rad(a.elevation), scale, a.size, k, rpts, rv, lpts, lcol)
            frames.append(compose(l_img, r_img, titles, bar))
        base = out_dir / f"{slab.name}_{stem}_spin"
        frames[0].save(f"{base}_poster.png")
        files = write_movie(frames, str(base), a.fps)
        print(f"   wrote {', '.join(str(f) for f in files)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

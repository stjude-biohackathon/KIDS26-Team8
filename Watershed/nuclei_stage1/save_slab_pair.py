"""
Save whole-slab pictures with the RAW image on the left and the same view with the NUCLEI filled in on the right.

Like view_chunk_3d.py but for a whole ROI, and written to PNG so it needs no display and no napari -- it runs on
a cluster node. (For an interactive 3-D render use view_slab on your own computer.)

Each row of the picture is one view; every view is drawn raw | nuclei at the same physical scale:
    proj_xy   whole-depth maximum projection looking down the coarse axis (rows = y, columns = x)
    sec_xy    a thin section around the middle of the depth (--section-um thick), same orientation
    proj_xz   whole-width projection (rows = coarse axis, columns = x)
    proj_yz   whole-height projection (rows = coarse axis, columns = y)
In the nuclei panel each nucleus gets its own colour over a dimmed copy of the raw image. Where several nuclei lie
on one line of sight, the one with the brightest raw voxel is drawn (so a projection shows the nuclei you would
see, not a smear). Scale bar = 1 mm.

Reads  <slab>/raw_chunked/*.tif  and  <slab>/<labels-dir>/{chunk}_labels.tif   (same inputs as view_slab;
respects the .global_ids marker written by stitch.py / run_halo_stage1.py)
Writes <out>/<slab>_<roi>_pair.png   (default out = <slab>/figures)

Usage:
    python -m nuclei_stage1.save_slab_pair <slab_dir>                                    # lists ROIs, exits
    python -m nuclei_stage1.save_slab_pair <slab_dir> --roi all --labels-dir labels_stitched
    python -m nuclei_stage1.save_slab_pair <slab_dir> --roi 0 --views proj_xy,sec_xy,proj_xz --downsample 4
    python -m nuclei_stage1.save_slab_pair <slab_dir> --roi all --out figures --no-deskew     # native frame

Downsampling is plain striding (like view_slab), so at a large stride small nuclei thin out; the nucleus count
in the title is of nuclei still present in the downsampled picture, not the true total (see centroids/).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .config import Stage1Config
from .deskew import SLOPE, deskew_volume
from .chunks import list_chunks
from .view_slab import assemble, extent, parse_voxel_size, roi_groups

# (projection axis, section?) for each named view; rows/cols are the two remaining axes in array order
VIEWS = {"proj_xy": (0, False), "sec_xy": (0, True), "proj_xz": (1, False), "proj_yz": (2, False)}


def project(raw, lab, axis, lo=None, hi=None, block=48):
    """Collapse `axis` of (raw, lab). Returns (raw_max, label_of_brightest_labelled_voxel).
    Blocked along another axis so the temporary arrays stay small."""
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    raw, lab = raw[tuple(sl)], lab[tuple(sl)]
    keep = [a for a in range(3) if a != axis]
    rmax = np.zeros((raw.shape[keep[0]], raw.shape[keep[1]]), raw.dtype)
    lout = np.zeros(rmax.shape, lab.dtype)
    n0 = raw.shape[keep[0]]
    for s in range(0, n0, block):
        idx = [slice(None)] * 3
        idx[keep[0]] = slice(s, s + block)
        r, l = raw[tuple(idx)], lab[tuple(idx)]
        r = np.moveaxis(r, axis, -1) if axis != 2 else r          # put the projection axis last, then
        l = np.moveaxis(l, axis, -1) if axis != 2 else l          # the block's two other axes lead
        rmax[s:s + block] = r.max(axis=-1)
        score = np.where(l > 0, r.astype(np.int32), -1)
        pick = score.argmax(axis=-1)
        best = np.take_along_axis(l, pick[..., None], axis=-1)[..., 0]
        lout[s:s + block] = np.where(score.max(axis=-1) >= 0, best, 0)
    return rmax, lout


def colourise(lab):
    """Distinct colour per label id (golden-ratio hue), vectorised; 0 -> black."""
    from matplotlib.colors import hsv_to_rgb
    ids = lab.astype(np.int64)
    h = (ids * 0.61803398875) % 1.0
    s = 0.55 + 0.4 * ((ids * 7) % 5) / 4.0
    v = 0.80 + 0.2 * ((ids * 3) % 4) / 3.0
    rgb = hsv_to_rgb(np.stack([h, s, v], axis=-1))
    rgb[ids == 0] = 0
    return rgb


def stretch(img, lo_p=1.0, hi_p=99.8):
    f = img.astype(np.float32)
    nz = f[f > 0]
    if nz.size == 0:
        return np.zeros_like(f)
    lo, hi = np.percentile(nz, [lo_p, hi_p])
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def render_roi(raw, lab, pix_um, views, section_um, title, out_png, width_px, underlay=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z, Y, X = raw.shape
    rows = []
    for name in views:
        axis, is_section = VIEWS[name]
        lo = hi = None
        if is_section:
            half = max(1, int(round(section_um / pix_um[axis] / 2)))
            mid = raw.shape[axis] // 2
            lo, hi = max(0, mid - half), min(raw.shape[axis], mid + half)
        r, l = project(raw, lab, axis, lo, hi)
        keep = [a for a in range(3) if a != axis]
        h_um, w_um = r.shape[0] * pix_um[keep[0]], r.shape[1] * pix_um[keep[1]]
        sec = f" (section {(hi - lo) * pix_um[axis]:.0f} um)" if is_section else ""
        rows.append((name + sec, r, l, h_um, w_um))

    # figure geometry: two panels per row at the same um-per-inch
    gap = 0.03
    w_max = max(r[4] for r in rows)
    fig_w_in = width_px / 100.0
    panel_w_in = fig_w_in / (2 + gap * 3)
    in_per_um = panel_w_in / w_max
    heights = [r[3] * in_per_um + 0.35 for r in rows]
    fig = plt.figure(figsize=(fig_w_in, sum(heights) + 0.6), dpi=100, facecolor="black")
    fig.suptitle(title, color="white", fontsize=11, y=0.995)
    top = 1.0 - 0.45 / (sum(heights) + 0.6)
    for k, (name, r, l, h_um, w_um) in enumerate(rows):
        hh = heights[k] / (sum(heights) + 0.6)
        base = stretch(r)
        gray = np.stack([base] * 3, axis=-1)
        filled = gray * 0.35 if underlay else np.zeros_like(gray)
        m = l > 0
        filled[m] = colourise(l)[m]
        y0 = top - sum(heights[:k + 1]) / (sum(heights) + 0.6)
        for j, (img, lab_txt) in enumerate([(gray, "raw"), (filled, f"nuclei ({len(np.unique(l[m]))} visible)")]):
            x0 = (gap + j * (1 + gap)) / (2 + gap * 3)
            ax = fig.add_axes([x0, y0, w_um * in_per_um / fig_w_in, h_um * in_per_um / (sum(heights) + 0.6)])
            ax.imshow(img, extent=[0, w_um, h_um, 0], interpolation="antialiased", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.set_title(f"{name} | {lab_txt}", color="white", fontsize=9, pad=3)
            if j == 1 and w_um > 2000:
                ax.plot([w_um - 1150, w_um - 150], [h_um - 120, h_um - 120], color="white", lw=3)
                ax.text(w_um - 650, h_um - 200, "1 mm", color="white", ha="center", va="bottom", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="black")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir")
    ap.add_argument("--roi", default=None, help="ROI index, part of its name, or 'all'")
    ap.add_argument("--voxel-size", default=None, help="voxel size in um as z,y,x for display (default: config.voxel_size). Use to test a different value, e.g. 2.0,0.5616,0.84")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--downsample", type=int, default=4, help="stride in y and x (default 4)")
    ap.add_argument("--z-downsample", type=int, default=1, help="stride along the coarse axis (default 1)")
    ap.add_argument("--views", default="proj_xy,sec_xy", help="comma list of: " + ", ".join(VIEWS))
    ap.add_argument("--section-um", type=float, default=6.0, help="thickness of the sec_xy section")
    ap.add_argument("--width-px", type=int, default=3200, help="total picture width")
    ap.add_argument("--out", default=None, help="output folder (default <slab>/figures)")
    ap.add_argument("--no-underlay", action="store_true", help="nuclei panel on black instead of dimmed raw")
    ap.add_argument("--no-deskew", action="store_true",
                    help="show the native (sheared) frame. Default: camera shear undone (display only), like view_slab --deskew")
    ap.add_argument("--max-gb", type=float, default=6.0)
    a = ap.parse_args()

    views = [v.strip() for v in a.views.split(",") if v.strip()]
    bad = [v for v in views if v not in VIEWS]
    if bad:
        sys.exit(f"unknown view(s) {bad}; choose from {list(VIEWS)}")
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
    for idx, (stem, cs) in sel:
        Z, Y, X = extent(cs)
        gb = (-(-Z // dz)) * (-(-Y // dxy)) * (-(-X // dxy)) * 6 / 1e9
        print(f"[{idx}] {stem}: {len(cs)} chunks, mosaic ~{gb:.1f} GB")
        if gb > a.max_gb:
            print(f"   skipped: > {a.max_gb} GB. Raise --downsample (or --max-gb).")
            continue
        raw, lab = assemble(cs, slab, a.labels_dir, dz, dxy)
        pu = pix_um.copy()
        if not a.no_deskew:
            sl = SLOPE * dz / dxy
            raw, _ = deskew_volume(raw, slope=sl)
            lab, _ = deskew_volume(lab.astype(np.int32), is_labels=True, slope=sl)
            lab = lab.astype(np.uint32)
        out_png = out_dir / f"{slab.name}_{stem}_pair.png"
        title = (f"{slab.name} / {stem}   labels: {a.labels_dir}   stride z{dz} y/x{dxy}"
                 f"   voxel ({', '.join(f'{float(v):g}' for v in vs)}) um" + ("" if a.no_deskew else "   deskewed (display)"))
        render_roi(raw, lab, pu, views, a.section_um, title, out_png, a.width_px, underlay=not a.no_underlay)
        print(f"   wrote {out_png}")
        del raw, lab


if __name__ == "__main__":
    main()

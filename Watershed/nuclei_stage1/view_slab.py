"""
View a slab (one ROI, or all ROIs side by side) in napari: raw mosaic + the stage-1 labels, placed at their
true positions, downsampled so it fits in memory.

Chunks are 128^3 tiles named {roi_stem}_z{Z}_y{Y}_x{X}.tif where (Z,Y,X) is the offset inside ONE ROI
file, so different ROIs of a slab overlap in coordinates -> they are shown as separate layers (--roi N for one,
--roi all for every ROI laid out side by side).

Reads  <slab>/raw_chunked/*.tif      and   <slab>/labels/<name>_labels.tif   (written by run_pipeline_hpc.py)
       Use --labels-dir labels_stitched to view nuclei merged across chunk boundaries (see stitch.py).

Usage (run where the slab is readable -- on the HPC node with a display, or after copying the slab,
or a sub-folder of it, to your computer):
    python -m nuclei_stage1.view_slab <slab_dir>                      # lists the ROIs and their sizes, then exits
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0              # first ROI, default downsample
    python -m nuclei_stage1.view_slab <slab_dir> --roi all --points   # EVERY ROI of the slab in one window (side by side)
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0 --downsample 4 --z-downsample 1
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0 --points     # centroid dots only (very light, whole ROI)
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0 --deskew     # display-only: undo the camera shear so nuclei look upright
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0 --labels-dir labels_prev_20261008_101500   # an older run

BIG SLABS: build the downsampled mosaic where the data is (no display needed), copy only that, view it locally:
    python -m nuclei_stage1.view_slab <slab_dir> --roi 0 --downsample 4 --save-mosaic slab21_roi0     # on the HPC
    python -m nuclei_stage1.view_slab --open slab21_roi0                                              # on your computer
(--save-mosaic writes <prefix>_raw.tif, <prefix>_labels.tif and <prefix>_meta.json; with --roi all it writes
<prefix>_roi0_*, <prefix>_roi1_*, ... and `--open <prefix>` shows them all together.)
--roi all lays the ROIs out side by side with a gap: file names carry no stage position, so this is NOT their
true relative placement. --max-gb applies per ROI.

Downsampling is plain striding (every Nth voxel), so small nuclei get thin when it is large. Use
--points for an overview of the whole ROI, then re-run on a region with --zmin/--zmax/--ymin/... for detail.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

from .deskew import SLOPE, deskew_volume
from .chunks import list_chunks

CH = 128    # chunk edge, voxels


def parse_voxel_size(arg, default):
    """'z,y,x' in micrometres -> float array; None -> default (config.voxel_size)."""
    import numpy as _np
    if not arg:
        return _np.array(default, float)
    v = [float(t) for t in arg.split(",")]
    if len(v) != 3 or min(v) <= 0:
        raise SystemExit(f"--voxel-size needs three positive numbers z,y,x, got {arg!r}")
    return _np.array(v, float)


def roi_groups(chunks):
    g = {}
    for c in chunks:
        g.setdefault(c.roi_stem, []).append(c)
    return sorted(g.items())


def extent(cs):
    return (max(c.z for c in cs) + CH, max(c.y for c in cs) + CH, max(c.x for c in cs) + CH)


def _report_missing_labels(slab, labels_dir, n_found, n_total):
    """assemble() used to skip missing label files silently, giving an empty label volume (and a picture with
    0 nuclei). Say so loudly, and list the label folders that DO exist."""
    slab = Path(slab)
    alts = []
    for d in sorted(slab.iterdir()):
        if d.is_dir() and d.name.startswith("labels"):
            alts.append(f"    {d.name}/  ({len(list(d.glob('*_labels.tif')))} label files)")
    listing = "\n".join(alts) if alts else "    (no labels* folders in the slab at all -- has stage 1 been run on it?)"
    if n_found == 0:
        sys.exit(f"ERROR: no label files found in {slab / labels_dir}/ for this ROI (0 of {n_total} chunks).\n"
                 f"Label folders in {slab}:\n{listing}\nPass --labels-dir <one of these>. (A new run moves the old "
                 f"labels/ aside as labels_prev_<time>/.)")
    if n_found < n_total:
        print(f"  WARNING: only {n_found} of {n_total} chunks have label files in {labels_dir}/ -- the rest show no nuclei")


def assemble(cs, slab, labels_dir, dz, dxy, want_raw=True, want_labels=True):
    """Return (raw_mosaic, label_mosaic) as strided arrays. Label ids are made unique across chunks."""
    Z, Y, X = extent(cs)
    shape = (-(-Z // dz), -(-Y // dxy), -(-X // dxy))
    raw = np.zeros(shape, np.uint16) if want_raw else None
    lab = np.zeros(shape, np.uint32) if want_labels else None
    next_id = 0
    n_found = 0
    global_ids = (Path(slab) / labels_dir / ".global_ids").exists()   # written by stitch.py
    for i, c in enumerate(cs):
        sl = (slice(c.z // dz, c.z // dz + -(-CH // dz)),
              slice(c.y // dxy, c.y // dxy + -(-CH // dxy)),
              slice(c.x // dxy, c.x // dxy + -(-CH // dxy)))
        if want_raw:
            r = tifffile.imread(c.raw_tif)[::dz, ::dxy, ::dxy]
            raw[sl[0].start:sl[0].start + r.shape[0], sl[1].start:sl[1].start + r.shape[1],
                sl[2].start:sl[2].start + r.shape[2]] = r.astype(np.uint16, copy=False)
        if want_labels:
            lp = Path(slab) / labels_dir / f"{c.name}_labels.tif"
            if lp.exists():
                n_found += 1
                l = tifffile.imread(lp)[::dz, ::dxy, ::dxy].astype(np.uint32)
                if not global_ids:        # stitched labels already carry unique ids; per-chunk ones don't
                    l[l > 0] += next_id
                next_id = int(max(next_id, l.max()))
                lab[sl[0].start:sl[0].start + l.shape[0], sl[1].start:sl[1].start + l.shape[1],
                    sl[2].start:sl[2].start + l.shape[2]] = l
        if (i + 1) % 100 == 0:
            print(f"  read {i + 1}/{len(cs)} chunks")
    if want_labels:
        _report_missing_labels(slab, labels_dir, n_found, len(cs))
    return raw, lab


def load_points(cs, slab, cent_dir):
    import pandas as pd
    pts = []
    for c in cs:
        p = Path(slab) / cent_dir / f"{c.name}_centroids.csv"
        if p.exists():
            df = pd.read_csv(p)
            if len(df):
                # centroids were stored as chunk-local + offset (z,y,x) of this ROI already
                pts.append(df[["z", "y", "x"]].to_numpy(float))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab_dir", nargs="?")
    ap.add_argument("--save-mosaic", default=None, metavar="PREFIX",
                    help="build the mosaic, write it to PREFIX_*.tif and exit (no napari needed)")
    ap.add_argument("--open", default=None, metavar="PREFIX", help="open a mosaic saved with --save-mosaic")
    ap.add_argument("--roi", default=None, help="ROI index (from the listing) or part of its name")
    ap.add_argument("--voxel-size", default=None, help="voxel size in um as z,y,x for display (default: config.voxel_size). Use to test a different value, e.g. 2.0,0.5616,0.84")
    ap.add_argument("--downsample", type=int, default=4, help="stride in y and x (default 4)")
    ap.add_argument("--z-downsample", type=int, default=1, help="stride in z (default 1)")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--centroids-dir", default="centroids")
    ap.add_argument("--points", action="store_true", help="show centroid dots only (no image mosaic)")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--deskew", action="store_true",
                    help="DISPLAY ONLY: undo the nuclei camera's -7 deg shear (X += 0.1228*Z) in the mosaic / "
                         "centroids so nuclei look upright. The pipeline's labels are still in the native frame.")
    ap.add_argument("--zmin", type=int, default=0); ap.add_argument("--zmax", type=int, default=10**9)
    ap.add_argument("--ymin", type=int, default=0); ap.add_argument("--ymax", type=int, default=10**9)
    ap.add_argument("--xmin", type=int, default=0); ap.add_argument("--xmax", type=int, default=10**9)
    ap.add_argument("--max-gb", type=float, default=6.0, help="refuse to build a mosaic bigger than this")
    a = ap.parse_args()

    GAP_UM = 60.0      # empty space between ROIs when several are shown together

    def add_roi_layers(viewer, tag, raw, lab, sc, x_off_um, deskewed_note=""):
        tr = (0.0, 0.0, x_off_um)
        if raw is not None:
            viewer.add_image(raw, name=f"raw {tag}", colormap="gray", rendering="mip", opacity=0.5,
                             scale=sc, translate=tr)
        viewer.add_labels(lab, name=f"nuclei {tag}", rendering="iso_categorical",
                          iso_gradient_mode="smooth", scale=sc, translate=tr)

    if a.open:
        import json
        import napari
        from .config import Stage1Config
        vs0 = parse_voxel_size(a.voxel_size, Stage1Config().voxel_size)
        metas = [a.open] if Path(f"{a.open}_meta.json").exists() else \
            sorted(str(p)[:-len("_meta.json")] for p in Path(a.open).parent.glob(Path(a.open).name + "_roi*_meta.json"))
        if not metas:
            sys.exit(f"no {a.open}_meta.json or {a.open}_roi*_meta.json found")
        viewer = napari.Viewer(ndisplay=3, title=a.open)
        x_off = 0.0
        for pre in metas:
            meta = json.load(open(f"{pre}_meta.json"))
            vs_m = parse_voxel_size(a.voxel_size, meta["voxel_size"]) if (a.voxel_size or meta.get("voxel_size")) else vs0
            sc = vs_m * np.array([meta["dz"], meta["dxy"], meta["dxy"]])
            raw = tifffile.imread(f"{pre}_raw.tif") if Path(f"{pre}_raw.tif").exists() else None
            lab = tifffile.imread(f"{pre}_labels.tif")
            add_roi_layers(viewer, Path(pre).name, raw, lab, sc, x_off)
            x_off += lab.shape[2] * sc[2] + GAP_UM
            print(f"opened {pre} (strides z{meta['dz']} y/x{meta['dxy']}, scale {tuple(sc)})")
        napari.run()
        sys.exit(0)
    if not a.slab_dir:
        sys.exit("slab_dir is required (or use --open PREFIX)")
    chunks = list_chunks(a.slab_dir)
    groups = roi_groups(chunks)
    if not groups:
        sys.exit(f"no chunks found in {a.slab_dir}/raw_chunked")
    if a.roi is None:
        print(f"{len(chunks)} chunks, {len(groups)} ROI(s):")
        for i, (stem, cs) in enumerate(groups):
            Z, Y, X = extent(cs)
            print(f"  [{i}] {stem}: {len(cs)} chunks, extent z{Z} y{Y} x{X} voxels")
        sys.exit("\nre-run with --roi <index>, or --roi all to show every ROI side by side")
    if a.roi == "all":
        sel = list(enumerate(groups))
    else:
        sel = [(i, g) for i, g in enumerate(groups) if str(i) == a.roi] or \
              [(i, g) for i, g in enumerate(groups) if a.roi in g[0]]
        sel = sel[:1]
    if not sel:
        sys.exit(f"no ROI matches {a.roi!r}")
    if a.points and a.save_mosaic:
        sys.exit("--points and --save-mosaic can't be combined")

    from .config import Stage1Config
    vs = parse_voxel_size(a.voxel_size, Stage1Config().voxel_size)
    viewer = None
    if not a.save_mosaic:
        import napari
        title = f"{Path(a.slab_dir).name} / " + ("all ROIs" if a.roi == "all" else sel[0][1][0])
        viewer = napari.Viewer(ndisplay=3, title=title)

    x_off = 0.0
    total = 0
    for idx, (stem, cs) in sel:
        cs = [c for c in cs if a.zmin <= c.z < a.zmax and a.ymin <= c.y < a.ymax and a.xmin <= c.x < a.xmax]
        if not cs:
            print(f"[{idx}] {stem}: no chunks inside the requested box, skipped")
            continue
        tag = f"[{idx}] {stem}"
        if a.points:
            pts = load_points(cs, a.slab_dir, a.centroids_dir)
            if a.deskew and len(pts):
                pts = pts.copy(); pts[:, 2] += SLOPE * pts[:, 0]      # same map as deskew.to_deskewed_points
            print(f"{tag}: {len(pts)} centroids from {len(cs)} chunks")
            if len(pts):
                viewer.add_points(pts, name=f"centroids {tag}", size=8 / vs.min(), face_color="yellow",
                                  scale=vs, translate=(0.0, 0.0, x_off), out_of_slice_display=True)
            x_off += extent(cs)[2] * vs[2] + GAP_UM
            total += len(pts)
            continue
        dz, dxy = a.z_downsample, a.downsample
        Z, Y, X = extent(cs)
        gb = (-(-Z // dz)) * (-(-Y // dxy)) * (-(-X // dxy)) * (2 + 4) / 1e9
        print(f"{tag}: mosaic {-(-Z // dz)} x {-(-Y // dxy)} x {-(-X // dxy)} voxels (~{gb:.1f} GB)")
        if gb > a.max_gb:
            sys.exit(f"too big (> {a.max_gb} GB): raise --downsample, restrict with --zmin/--ymin/..., "
                     f"use --points, or pass --max-gb")
        raw, lab = assemble(cs, a.slab_dir, a.labels_dir, dz, dxy, want_raw=not a.no_raw)
        if a.deskew:
            sl = SLOPE * dz / dxy          # shear in MOSAIC index units (z strided by dz, x by dxy)
            if raw is not None:
                raw, _ = deskew_volume(raw, slope=sl)
            lab, _ = deskew_volume(lab.astype(np.int32), is_labels=True, slope=sl)
            lab = lab.astype(np.uint32)
            print(f"  deskewed for display (slope {sl:.4f} in mosaic units); X axis widened to {lab.shape[2]}")
        if a.save_mosaic:
            import json
            pre = a.save_mosaic if a.roi != "all" else f"{a.save_mosaic}_roi{idx}"
            if raw is not None:
                tifffile.imwrite(f"{pre}_raw.tif", raw, compression="zlib")
            tifffile.imwrite(f"{pre}_labels.tif", lab, compression="zlib")
            json.dump({"dz": dz, "dxy": dxy, "roi": stem, "n_labels": int(lab.max()), "deskewed": bool(a.deskew),
                       "voxel_size": [float(v) for v in vs]},
                      open(f"{pre}_meta.json", "w"))
            print(f"  saved {pre}_raw.tif / _labels.tif / _meta.json  ({int(lab.max())} nuclei)")
            continue
        sc = vs * np.array([dz, dxy, dxy])
        add_roi_layers(viewer, tag, raw, lab, sc, x_off)
        x_off += lab.shape[2] * sc[2] + GAP_UM
        total += int(lab.max())
        print(f"  {int(lab.max())} labelled nuclei (ids unique across chunks)")
        del raw, lab
    if a.save_mosaic:
        sys.exit(0)
    print(f"{total} {'centroids' if a.points else 'labelled nuclei'} in view. "
          "Viewer open -- drag to rotate, scroll to zoom." +
          ("  (ROIs are laid out side by side, NOT at true relative positions.)" if len(sel) > 1 else ""))
    napari.run()


if __name__ == "__main__":
    main()

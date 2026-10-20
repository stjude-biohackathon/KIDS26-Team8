"""
Quality-control report for a finished stage-1 run on one volume (a *_chunks folder with raw_chunked/ and labels/).

Run it after the segmentation (the LSF script does this automatically):
    python -m nuclei_stage1.qc <volume_dir> [--workers 8]

Writes <volume_dir>/qc/ :
    qc_report.html        one self-contained page: summary, ranked review list, representative images
                          (ALWAYS at least one image from the edge of the volume), audit instructions
    qc_chunks.csv         every chunk: status, reasons, descriptors, nuclei counts and sizes, neighbours
    qc_review_list.csv    only the chunks that need a look, most important first
    qc_audit_sample.csv   a RANDOM sample of chunks for a human to judge (never overwritten once it exists)
    qc_summary.json       the headline numbers
    images/*.png          the images shown in the report

What it checks (every flag is a heuristic, none is a guarantee -- see "Limits" in the report):
    missing_output           a raw chunk has no label file (the run did not finish / crashed on it)
    unlike_exemplars         the chunk looks unlike every ground-truthed exemplar (same test as --adaptive), so the
                             settings used were never validated on anything like it
    no_nuclei_but_signal     zero nuclei found although the raw data has a bright tail
    density_jump_vs_neighbors  nuclei per megavoxel differs >3x from the median of the neighbouring chunks
    oversized_outlier / tiny_outlier / size_shift_outlier / ratio_outlier
                             the chunk's fraction of very large / very small nuclei, typical nucleus size, or
                             (nuclei found)/(bright blobs) is an outlier (robust z > 4) compared with the REST OF
                             THIS VOLUME -- self-calibrating, so it needs no ground truth
    Status: REVIEW (look at it), WATCH (one minor reason), OK, PADDING (mostly zero padding, nothing to segment),
            MISSING (no output).

The audit sample is the only thing that tells you the real error rate: judge the sampled chunks (fill in the
`verdict` column with ok / minor / major) and re-run this script; the report then shows the fraction judged ok with
a 95% interval.  Flags are triage; the audit is the measurement.
"""
import argparse
import base64
import html
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from .autoconfig import EXEMPLAR_PATH, MATCH_KEYS, load_exemplars
from .chunks import list_chunks
from .config import Stage1Config
from .descriptors import describe_chunk

CHUNK = 128
WEIGHTS = {"missing_output": 10, "unlike_exemplars": 5, "no_nuclei_but_signal": 5,
           "density_jump_vs_neighbors": 3, "oversized_outlier": 3, "tiny_outlier": 2,
           "size_shift_outlier": 2, "ratio_outlier": 2}
ROBUST_Z = 4.0


# ---------------------------------------------------------------- per-chunk measurements
def _measure(job):
    chunk, labels_dir, exemplars, max_distance, cfg = job
    row = {"chunk": chunk.name, "roi": chunk.roi_stem, "z": chunk.z, "y": chunk.y, "x": chunk.x}
    lab_path = Path(labels_dir) / f"{chunk.name}_labels.tif"
    row["has_output"] = lab_path.exists()
    raw = tifffile.imread(chunk.raw_tif)
    d = describe_chunk(raw, cfg.nucleus_diameter_vox)
    row["zero_frac"] = d["zero_frac"]
    row["usable"] = bool(d.get("usable"))
    for k in ("contrast_p99", "fg_frac_k3", "blobs_per_mvox", "n_blobs"):
        row[k] = d.get(k, np.nan)
    row["exemplar_distance"], row["nearest_exemplar"] = np.nan, ""
    if d.get("usable") and exemplars:
        v = np.log(np.array([max(d[k], 1e-6) for k in MATCH_KEYS]))
        dists = [float(np.linalg.norm(v - np.array(e["match_vec"]))) for e in exemplars]
        i = int(np.argmin(dists))
        row["exemplar_distance"], row["nearest_exemplar"] = dists[i], exemplars[i]["name"]
    if not row["has_output"]:
        return row
    labels = tifffile.imread(lab_path)
    valid_vox = max(int((raw > 0).sum()), 1)
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sizes[sizes > 0]
    row["n_nuclei"] = int(len(sizes))
    row["nuclei_per_mvox"] = len(sizes) / (valid_vox / 1e6)
    row["labelled_frac"] = float((labels > 0).sum() / labels.size)
    expected = (4 / 3) * math.pi * (cfg.nucleus_diameter_vox / 2) ** 3
    if len(sizes):
        row["median_vol_vox"] = float(np.median(sizes))
        row["oversized_frac"] = float((sizes > 3 * expected).mean())
        row["tiny_frac"] = float((sizes < 0.2 * expected).mean())
        objs = [o for o in ndi.find_objects(labels) if o is not None]
        touch = sum(any(s.start == 0 or s.stop == n for s, n in zip(o, labels.shape)) for o in objs)
        row["border_touching_frac"] = touch / max(len(objs), 1)
        row["ratio_to_blobs"] = len(sizes) / d["n_blobs"] if d.get("n_blobs") else np.nan
    return row


# ---------------------------------------------------------------- flags
def _robust_outlier(s, floor, two_sided=False, log=False):
    """Boolean Series: robust z > ROBUST_Z against the rest of the volume (NaNs never flag)."""
    x = np.log(s.clip(lower=1e-9)) if log else s
    ok = x.notna()
    if ok.sum() < 8:
        return pd.Series(False, index=s.index)
    med = x[ok].median()
    scale = max(1.4826 * (x[ok] - med).abs().median(), floor)
    z = (x - med) / scale
    return (z.abs() if two_sided else z) > ROBUST_Z


def _neighbour_density_flag(df, step=CHUNK):
    key = {(r.roi, r.z, r.y, r.x): r.nuclei_per_mvox for r in df.itertuples() if r.usable and r.has_output}
    flag, n_nb = [], []
    for r in df.itertuples():
        nb = [key[(r.roi, r.z + dz * step, r.y + dy * step, r.x + dx * step)]
              for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dz, dy, dx) != (0, 0, 0) and (r.roi, r.z + dz * step, r.y + dy * step, r.x + dx * step) in key]
        n_nb.append(len(nb))
        if not (r.usable and r.has_output) or len(nb) < 3:
            flag.append(False)
            continue
        m = float(np.median(nb))
        mine = r.nuclei_per_mvox
        flag.append(bool(max(m, mine) >= 2.0 and (mine > 3 * max(m, 0.5) or mine < m / 3)))
    return pd.Series(flag, index=df.index), pd.Series(n_nb, index=df.index)


def _edge_info(df):
    """Number of lateral (y, x) faces of its ROI that a chunk touches: 0 interior, 1 edge, 2 corner. z is not
    counted: a typical ROI is only 2 chunks deep, so every chunk would be a z-edge and the word would mean nothing."""
    lo = df.groupby("roi")[["y", "x"]].transform("min")
    hi = df.groupby("roi")[["y", "x"]].transform("max")
    return sum((df[a] == lo[a]).astype(int) | (df[a] == hi[a]).astype(int) for a in ("y", "x"))


def assess(df, has_exemplars, max_distance, step=CHUNK):
    n_edge = _edge_info(df)
    df["edge_axes"] = n_edge
    df["is_edge"] = n_edge > 0
    dens, n_nb = _neighbour_density_flag(df, step)
    df["n_neighbours"] = n_nb
    f = pd.DataFrame(index=df.index)
    f["missing_output"] = ~df["has_output"]
    f["unlike_exemplars"] = (df["usable"] & df["exemplar_distance"].gt(max_distance)) if has_exemplars else False
    f["no_nuclei_but_signal"] = (df["usable"] & df["has_output"] & df["n_nuclei"].eq(0)
                                 & df["contrast_p99"].gt(3.0))
    f["density_jump_vs_neighbors"] = dens
    base = df["usable"] & df["has_output"]
    f["oversized_outlier"] = base & _robust_outlier(df["oversized_frac"], 0.05)
    f["tiny_outlier"] = base & _robust_outlier(df["tiny_frac"], 0.05)
    f["size_shift_outlier"] = base & _robust_outlier(df["median_vol_vox"], 0.15, two_sided=True, log=True)
    f["ratio_outlier"] = base & _robust_outlier(df["ratio_to_blobs"], 0.3, two_sided=True)
    f = f.fillna(False).astype(bool)
    df["priority"] = sum(f[c].astype(int) * w for c, w in WEIGHTS.items())
    df["reasons"] = f.apply(lambda r: ";".join(c for c in WEIGHTS if r[c]), axis=1)
    n_reasons = f.sum(axis=1)
    status = np.where(~df["has_output"], "MISSING",
             np.where(~df["usable"], "PADDING",
             np.where((df["priority"] >= 5) | (n_reasons >= 2), "REVIEW",
             np.where(n_reasons == 1, "WATCH", "OK"))))
    df["status"] = status
    return df


# ---------------------------------------------------------------- audit sample
def audit_sample(df, out_csv, frac, n_min, seed):
    out_csv = Path(out_csv)
    if out_csv.exists():
        return pd.read_csv(out_csv)
    pool = df[df["usable"] & df["has_output"]]
    n = min(len(pool), max(n_min, math.ceil(frac * len(pool))))
    s = pool.sample(n=n, random_state=seed).sort_values(["roi", "z", "y", "x"])
    s = s[["chunk", "roi", "z", "y", "x", "status", "reasons"]].copy()
    s["verdict"], s["reviewer"], s["notes"] = "", "", ""
    s.to_csv(out_csv, index=False)
    return s


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def audit_result(aud):
    v = aud["verdict"].fillna("").astype(str).str.strip().str.lower()
    done = v[v.isin(["ok", "minor", "major"])]
    n = len(done)
    if n == 0:
        return None
    k = int((done == "ok").sum())
    lo, hi = wilson(k, n)
    return {"judged": n, "of": len(aud), "ok": k, "minor": int((done == "minor").sum()),
            "major": int((done == "major").sum()), "ok_frac": k / n, "ci95": [lo, hi]}


# ---------------------------------------------------------------- images
def _pick_images(df, n_edge, n_interior, n_flagged, seed):
    """Images for the MAIN report: edge chunks first (always at least one when the volume has an edge), a few random
    interior chunks, and only the n_flagged highest-priority flagged chunks. Everything else goes to the detailed report."""
    rng = np.random.default_rng(seed)
    good = df[df["usable"] & df["has_output"] & (df["n_nuclei"] > 0)]
    picks = []                                                   # (chunk name, why)

    def add(name, why):
        if name not in [p[0] for p in picks]:
            picks.append((name, why))

    edge = good[good["is_edge"]]
    if len(edge):
        corners = edge[edge["edge_axes"] >= 2]
        order = list(corners.sample(frac=1, random_state=seed)["chunk"][:1])
        rest = edge[~edge["chunk"].isin(order)].sample(frac=1, random_state=seed + 1)
        rois_seen = set(edge[edge["chunk"].isin(order)]["roi"])
        for r in rest.itertuples():                              # spread the edge picks over ROIs first
            if len(order) >= n_edge:
                break
            if r.roi not in rois_seen or len(rois_seen) >= edge["roi"].nunique():
                order.append(r.chunk)
                rois_seen.add(r.roi)
        for name in order[:n_edge]:
            add(name, "edge of volume")
    interior = good[~good["is_edge"]]
    if len(interior):
        for name in rng.choice(interior["chunk"].values, size=min(n_interior, len(interior)), replace=False):
            add(name, "random interior")
    flagged = df[df["status"].isin(["REVIEW", "MISSING"]) & df["has_output"]].sort_values("priority", ascending=False)
    for r in flagged.head(n_flagged).itertuples():
        add(r.chunk, f"flagged: {r.reasons.replace(';', ', ')}")
    return picks


def _pick_detail_images(df, main_names, max_n):
    """Extra images for the DETAILED report: the remaining REVIEW chunks by priority (capped, to keep the report light),
    then the densest and sparsest chunks that contain nuclei."""
    picks = []
    flagged = df[(df["status"] == "REVIEW") & df["has_output"] & ~df["chunk"].isin(main_names)].sort_values(
        "priority", ascending=False)
    for r in flagged.head(max_n).itertuples():
        picks.append((r.chunk, f"flagged: {r.reasons.replace(';', ', ')}"))
    good = df[df["usable"] & df["has_output"] & (df["n_nuclei"] > 0)]
    if len(good):
        srt = good.sort_values("nuclei_per_mvox")
        for name, why in ((srt.iloc[-1]["chunk"], "densest chunk"), (srt.iloc[0]["chunk"], "sparsest chunk with nuclei")):
            if name not in main_names and name not in [p[0] for p in picks]:
                picks.append((name, why))
    return picks


def _render(chunk, labels_dir, voxel_size, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    raw = tifffile.imread(chunk.raw_tif).astype(np.float32)
    lab = tifffile.imread(Path(labels_dir) / f"{chunk.name}_labels.tif")
    fg = (lab > 0).sum(axis=(1, 2))
    z = int(np.argmax(fg)) if fg.max() > 0 else raw.shape[0] // 2
    y = int(np.argmax((lab > 0).sum(axis=(0, 2)))) if fg.max() > 0 else raw.shape[1] // 2
    nz = raw[raw > 0]
    lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99.7)) if nz.size else (0, 1)

    def gray(a):
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    rng = np.random.default_rng(1)
    palette = rng.random((int(lab.max()) + 2, 3)) * 0.7 + 0.3

    def overlay(img, l):
        rgb = np.dstack([gray(img)] * 3)
        b = find_boundaries(l, mode="inner")
        rgb[b] = palette[l[b]]
        return rgb

    vz, vy, vx = voxel_size
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 3.5), facecolor="white", gridspec_kw={"width_ratios": [1, 1, 0.42]})
    ax[0].imshow(gray(raw[z]), cmap="gray", interpolation="nearest"); ax[0].set_title(f"raw, z={z}", fontsize=9)
    ax[1].imshow(overlay(raw[z], lab[z]), interpolation="nearest"); ax[1].set_title(f"labels, z={z}", fontsize=9)
    ax[2].imshow(overlay(raw[:, y, :], lab[:, y, :]), interpolation="nearest", aspect=vz / vx)
    ax[2].set_title(f"labels, side view y={y}", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title if len(title) < 110 else title[:107] + "...", fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- html
def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


BADGE = {"OK": "#2e7d32", "WATCH": "#b26a00", "REVIEW": "#c62828", "MISSING": "#6a1b9a", "PADDING": "#607d8b"}
CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1b1b1b;background:#fcfcfb}
h1,h2{margin:.9em 0 .3em} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px} figure{margin:0} img{width:100%;border:1px solid #ddd}
figcaption{font-size:13px;color:#444} code{background:#eee;padding:1px 4px} .note{background:#fff6e0;border:1px solid #e8d08a;padding:8px 12px}
a{color:#1a5fb4} input,select{font:inherit;padding:3px 6px;margin-right:8px}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}td,th{border-color:#333}code{background:#2a2a2a}.note{background:#2b2616;border-color:#5a4a1a}img{border-color:#333}figcaption{color:#aaa}a{color:#78aeed}}"""


def _audit_html(audit, audit_res, vol):
    e = html.escape
    if audit_res:
        return (f"<p><b>Audit result:</b> {audit_res['ok']} of {audit_res['judged']} judged chunks were <b>ok</b> "
                f"({100 * audit_res['ok_frac']:.0f}%, 95% interval {100 * audit_res['ci95'][0]:.0f}-"
                f"{100 * audit_res['ci95'][1]:.0f}%); {audit_res['minor']} minor, {audit_res['major']} major "
                f"({audit_res['judged']} of {audit_res['of']} sampled chunks judged so far).</p>")
    return (f"<p><b>Audit not done yet.</b> Open <code>qc_audit_sample.csv</code> ({len(audit)} randomly chosen chunks), look at each one "
            f"(napari or the images), write <code>ok</code>, <code>minor</code> or <code>major</code> in the "
            f"<code>verdict</code> column, then re-run <code>python -m nuclei_stage1.qc {e(str(vol))}</code>. "
            f"This random sample - not the flags - is what measures the real error rate.</p>")


def write_html(path, vol, df, summary, picks, images, audit, audit_res, limits, detail_name="qc_detailed.html"):
    """The MAIN report: short, self-contained (images embedded), meant to be glanced at."""
    e = html.escape
    cards = "".join(
        f"<figure><img src='data:image/png;base64,{_b64(img)}'><figcaption><b>{e(why)}</b><br>{e(name)}</figcaption></figure>"
        for (name, why), img in zip(picks, images))
    cnt = summary["status_counts"]
    stat = " &nbsp; ".join(f"<span style='color:{BADGE[k]}'><b>{k}</b> {v}</span>" for k, v in cnt.items())
    rc = summary["reason_counts"]
    reasons = "".join(f"<tr><td>{e(k.replace('_', ' '))}</td><td>{v}</td></tr>" for k, v in rc.items()) or \
        "<tr><td colspan=2>no flags raised</td></tr>"
    timing = summary.get("timing")
    t = (f"<p>Run time: {timing['chunks']} chunks, median {timing['median_s']:.1f} s per chunk per worker.</p>" if timing else "")
    n_flag = summary["n_flagged"]
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>QC report - {e(Path(vol).name)}</title><style>{CSS}</style></head><body>
<h1>QC report - {e(Path(vol).name)}</h1>
<p>{summary['n_chunks']} chunks &nbsp;|&nbsp; output present for {summary['n_with_output']} ({100 * summary['coverage']:.1f}%) &nbsp;|&nbsp; {stat}</p>
<p>{summary['n_nuclei_total']:,} nuclei in total; median {summary['median_nuclei_per_mvox']:.1f} per megavoxel.
{summary['n_unlike']} chunk(s) look unlike every ground-truthed exemplar.</p>{t}
<h2>Representative segmentations</h2>
<p>Raw slice, the same slice with nucleus outlines, and a side view. The first images are always from the <b>edge</b> of the volume,
where tissue and padding make segmentation hardest; then random interior chunks and the {summary['n_main_flagged']} highest-priority flagged chunk(s).</p>
<div class='grid'>{cards}</div>
<h2>What the checks found</h2>
<table style='max-width:520px'><tr><th>check</th><th>chunks</th></tr>{reasons}</table>
<p>{n_flag} chunk(s) are flagged in total. <b><a href='{detail_name}'>Open the detailed report</a></b> for every flagged chunk,
with reasons, positions and more images (<code>{detail_name}</code>, same folder).</p>
<h2>Audit</h2>{_audit_html(audit, audit_res, vol)}
<h2>Limits - read before trusting this</h2><div class='note'>{limits}</div>
</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def write_detailed_html(path, vol, df, picks, image_names, n_main, main_name="qc_report.html"):
    """The DETAILED report: every flagged chunk in a filterable table, plus extra images (linked from images/, not
    embedded, so it stays light even with thousands of chunks)."""
    e = html.escape
    fl = df[df["status"].isin(["REVIEW", "MISSING", "WATCH"])].sort_values(["priority", "chunk"], ascending=[False, True])
    img_for = dict(zip([p[0] for p in picks], image_names))
    rows = []
    for r in fl.itertuples():
        link = f"<a href='images/{e(img_for[r.chunk])}'>image</a>" if r.chunk in img_for else ""
        nn = "" if pd.isna(r.n_nuclei) else int(r.n_nuclei)
        dens = "" if pd.isna(r.nuclei_per_mvox) else f"{r.nuclei_per_mvox:.1f}"
        dist = "" if pd.isna(r.exemplar_distance) else f"{r.exemplar_distance:.2f}"
        rows.append(f"<tr data-status='{r.status}' data-reasons='{e(r.reasons)}'><td>{e(r.chunk)}</td>"
                    f"<td><b style='color:{BADGE[r.status]}'>{r.status}</b></td><td>{e(r.reasons.replace(';', ', '))}</td>"
                    f"<td>{r.priority}</td><td>{nn}</td><td>{dens}</td><td>{dist}</td><td>{'edge' if r.is_edge else ''}</td><td>{link}</td></tr>")
    cards = "".join(
        f"<figure><a href='images/{e(img)}'><img loading='lazy' src='images/{e(img)}'></a>"
        f"<figcaption><b>{e(why)}</b><br>{e(name)}</figcaption></figure>"
        for (name, why), img in zip(picks[n_main:], image_names[n_main:]))
    reasons = sorted({x for r in fl["reasons"] for x in r.split(";") if x})
    opts = "".join(f"<option value='{e(x)}'>{e(x.replace('_', ' '))}</option>" for x in reasons)
    js = """const rows=[...document.querySelectorAll('#t tbody tr')];
function f(){const s=document.getElementById('s').value,r=document.getElementById('r').value,q=document.getElementById('q').value.toLowerCase();
let n=0;rows.forEach(x=>{const ok=(!s||x.dataset.status==s)&&(!r||x.dataset.reasons.split(';').includes(r))&&(!q||x.textContent.toLowerCase().includes(q));
x.style.display=ok?'':'none';if(ok)n++;});document.getElementById('n').textContent=n+' shown';}
['s','r','q'].forEach(i=>document.getElementById(i).addEventListener('input',f));f();"""
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Detailed QC - {e(Path(vol).name)}</title><style>{CSS}</style></head><body>
<h1>Detailed QC - {e(Path(vol).name)}</h1>
<p><a href='{main_name}'>Back to the main report</a> &nbsp;|&nbsp; {len(fl)} flagged chunk(s). Images are in <code>images/</code> next to this file;
the full table is also in <code>qc_review_list.csv</code> and every chunk in <code>qc_chunks.csv</code>.</p>
<h2>Every flagged chunk</h2>
<p><select id='s'><option value=''>any status</option><option>REVIEW</option><option>WATCH</option><option>MISSING</option></select>
<select id='r'><option value=''>any reason</option>{opts}</select>
<input id='q' placeholder='search chunk name'><span id='n'></span></p>
<table id='t'><thead><tr><th>chunk</th><th>status</th><th>why</th><th>priority</th><th>nuclei</th><th>per Mvox</th><th>exemplar dist.</th><th></th><th></th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>More images ({len(picks) - n_main})</h2>
<p>The remaining highest-priority REVIEW chunks (capped to keep this page light; the table above lists all of them) and the
densest and sparsest chunks. The images shown in the main report are linked from the table.</p>
<div class='grid'>{cards}</div>
<script>{js}</script></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


LIMITS = ("The flags are heuristics and are not calibrated against ground truth (we have very few corrected chunks). "
          "A chunk can pass every check and still be segmented badly, and a flagged chunk can be fine. "
          "'Unlike exemplars' only compares three image statistics. The outlier tests compare each chunk with the rest of "
          "<i>this</i> volume, so they cannot notice a problem that affects the whole volume. "
          "Image scale uses the configured voxel size, which has not been verified against the microscope metadata. "
          "Only the random audit gives a real error rate.")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume_dir")
    ap.add_argument("--labels-dir", default="labels", help="folder (inside the volume) with <chunk>_labels.tif")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-distance", type=float, default=0.75, help="exemplar distance above which a chunk is 'unlike'")
    ap.add_argument("--exemplars", default=str(EXEMPLAR_PATH))
    ap.add_argument("--n-edge", type=int, default=3, help="edge chunks to show (at least 1 whenever the volume has an edge)")
    ap.add_argument("--n-interior", type=int, default=3)
    ap.add_argument("--n-flagged", type=int, default=2, help="flagged chunks shown in the MAIN report (the rest go to the detailed one)")
    ap.add_argument("--max-detail-images", type=int, default=150, help="most extra images rendered for the detailed report")
    ap.add_argument("--audit-frac", type=float, default=0.015)
    ap.add_argument("--audit-min", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=CHUNK, help="edge length of a chunk (voxels); sets the neighbour grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="qc", help="output folder name inside the volume")
    a = ap.parse_args(argv)

    vol = Path(a.volume_dir)
    out = vol / a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    chunks = list_chunks(vol)
    if not chunks:
        sys.exit(f"no chunks in {vol / 'raw_chunked'}")
    cfg = Stage1Config()
    exemplars = load_exemplars(a.exemplars)
    labels_dir = vol / a.labels_dir
    jobs = [(c, labels_dir, exemplars, a.max_distance, cfg) for c in chunks]
    print(f"QC of {len(chunks)} chunks in {vol.name} ({a.workers} worker(s)) ...", flush=True)
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_measure, jobs, chunksize=4))
    else:
        rows = [_measure(j) for j in jobs]
    df = pd.DataFrame(rows)
    for col in ("n_nuclei", "nuclei_per_mvox", "labelled_frac", "median_vol_vox", "oversized_frac", "tiny_frac",
                "border_touching_frac", "ratio_to_blobs"):
        if col not in df:
            df[col] = np.nan
    df = assess(df, bool(exemplars), a.max_distance, a.chunk_size)
    df.sort_values(["roi", "z", "y", "x"]).to_csv(out / "qc_chunks.csv", index=False)
    flagged = df[df["status"].isin(["REVIEW", "MISSING", "WATCH"])].sort_values("priority", ascending=False)
    flagged[["chunk", "status", "reasons", "priority", "n_nuclei", "nuclei_per_mvox", "exemplar_distance",
             "is_edge"]].to_csv(out / "qc_review_list.csv", index=False)
    review = flagged[flagged["status"].isin(["REVIEW", "MISSING"])]

    aud = audit_sample(df, out / "qc_audit_sample.csv", a.audit_frac, a.audit_min, a.seed)
    audit_res = audit_result(aud)

    by_name = {c.name: c for c in chunks}
    for old in (out / "images").glob("*.png"):          # stale images from an earlier run
        old.unlink()

    def render_all(plist, prefix):
        """Returns (picks that rendered, their png paths). A failing image (for example matplotlib missing) is skipped
        with a warning instead of losing the whole report."""
        ok, pngs = [], []
        for i, (name, why) in enumerate(plist):
            r = df[df["chunk"] == name].iloc[0]
            title = f"{name}   [{why}]   {int(r.n_nuclei)} nuclei, {r.nuclei_per_mvox:.1f}/Mvox, status {r.status}"
            png = out / "images" / f"{prefix}{i:03d}_{name}.png"
            try:
                _render(by_name[name], labels_dir, cfg.voxel_size, title, png)
            except Exception as exc:
                print(f"WARNING: could not render an image for {name}: {exc!r}", flush=True)
                continue
            ok.append((name, why)); pngs.append(png)
        return ok, pngs

    picks, images = render_all(_pick_images(df, a.n_edge, a.n_interior, a.n_flagged, a.seed), "main_")
    detail_picks, detail_images = render_all(
        _pick_detail_images(df, {p[0] for p in picks}, a.max_detail_images), "detail_")

    reason_counts = {}
    for r in df["reasons"]:
        for x in r.split(";"):
            if x:
                reason_counts[x] = reason_counts.get(x, 0) + 1
    summary = {
        "volume": vol.name, "n_chunks": int(len(df)), "n_with_output": int(df["has_output"].sum()),
        "coverage": float(df["has_output"].mean()),
        "status_counts": {k: int(v) for k, v in df["status"].value_counts().reindex(
            ["OK", "WATCH", "REVIEW", "MISSING", "PADDING"]).fillna(0).items() if v},
        "n_flagged": int(len(flagged)),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "n_nuclei_total": int(df["n_nuclei"].fillna(0).sum()),
        "median_nuclei_per_mvox": float(df.loc[df["usable"] & df["has_output"], "nuclei_per_mvox"].median()),
        "n_unlike": int(df["reasons"].str.contains("unlike_exemplars").sum()),
        "n_edge_images": int(sum(1 for _, w in picks if w == "edge of volume")),
        "n_main_flagged": int(sum(1 for _, w in picks if w.startswith("flagged"))),
        "audit": audit_res,
    }
    tp = vol / "timing_per_chunk.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        summary["timing"] = {"chunks": int(len(t)), "median_s": float(t["seconds"].median())}
    (out / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    limits = LIMITS if exemplars else (
        "<b>The 'unlike exemplars' check was SKIPPED: no exemplar file was found (" + html.escape(str(a.exemplars)) +
        ").</b> Reinstall the package (exemplars.json ships with it) to enable it. " + LIMITS)
    write_html(out / "qc_report.html", vol, df, summary, picks, images, aud, audit_res, limits)
    write_detailed_html(out / "qc_detailed.html", vol, df, picks + detail_picks,
                        [p.name for p in images + detail_images], len(picks))

    print(f"\n=== QC summary: {vol.name} ===")
    if not exemplars:
        print(f"WARNING: no exemplar file at {a.exemplars} -- the 'unlike exemplars' check was skipped")
    print(f"chunks: {summary['n_chunks']}, with output: {summary['n_with_output']} ({100 * summary['coverage']:.1f}%)")
    print("status: " + ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items()))
    print(f"nuclei: {summary['n_nuclei_total']:,}; {summary['n_unlike']} chunk(s) unlike every exemplar")
    if len(review):
        print("top chunks to look at:")
        for r in review.head(10).itertuples():
            print(f"  {r.status:<8}{r.chunk:<48}{r.reasons}")
    print("audit: " + (f"{audit_res['ok']}/{audit_res['judged']} judged ok" if audit_res
                       else f"not done -- judge the {len(aud)} chunks in {out / 'qc_audit_sample.csv'}"))
    print(f"report:   {out / 'qc_report.html'}  (edge images: {summary['n_edge_images']})")
    print(f"detailed: {out / 'qc_detailed.html'}  ({summary['n_flagged']} flagged chunk(s))")


if __name__ == "__main__":
    main()

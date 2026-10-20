"""QC report: flags a missing output, always shows an edge image, and never overwrites a filled-in audit sheet."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import label

from nuclei_stage1 import qc


def _make_volume(root, n=64):
    rng = np.random.default_rng(0)
    raw_dir, lab_dir = Path(root) / "raw_chunked", Path(root) / "labels"
    raw_dir.mkdir(); lab_dir.mkdir()
    zz, yy, xx = np.mgrid[:n, :n, :n]
    for iy in (0, 1):
        for ix in (0, 1):
            vol = rng.normal(3000, 300, (n, n, n))
            for _ in range(6):
                c = rng.integers(12, n - 12, 3)
                vol += 3500 * np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * 3.0 ** 2)))
            name = f"roi_z000000_y{iy * n:06d}_x{ix * n:06d}"
            tifffile.imwrite(raw_dir / f"{name}.tif", vol.astype(np.uint16))
            lab = label(ndi.gaussian_filter(vol, 1) > 4500).astype(np.uint32)
            tifffile.imwrite(lab_dir / f"{name}_labels.tif", lab)
    return lab_dir


def test_qc_report(tmp_path):
    lab_dir = _make_volume(tmp_path)
    missing = sorted(lab_dir.glob("*.tif"))[-1]
    missing.unlink()                                                # simulate a chunk that never got segmented
    args = [str(tmp_path), "--chunk-size", "64", "--n-interior", "1", "--audit-min", "2"]
    qc.main(args)
    out = tmp_path / "qc"
    for f in ("qc_report.html", "qc_detailed.html", "qc_chunks.csv", "qc_review_list.csv", "qc_audit_sample.csv", "qc_summary.json"):
        assert (out / f).exists(), f
    df = pd.read_csv(out / "qc_chunks.csv")
    assert (df["status"] == "MISSING").sum() == 1 and df.loc[df["status"] == "MISSING", "chunk"].iloc[0] == missing.name.replace("_labels.tif", "")
    summary = json.loads((out / "qc_summary.json").read_text())
    assert summary["n_edge_images"] >= 1                           # an edge chunk is always shown
    assert "edge of volume" in (out / "qc_report.html").read_text()
    assert missing.name.replace("_labels.tif", "") in (out / "qc_detailed.html").read_text()   # every flagged chunk is in the detailed report
    assert "qc_detailed.html" in (out / "qc_report.html").read_text()

    # a filled-in audit sheet survives a re-run and is reported
    aud = pd.read_csv(out / "qc_audit_sample.csv")
    aud["verdict"] = ["ok"] * len(aud)
    aud.to_csv(out / "qc_audit_sample.csv", index=False)
    qc.main(args)
    summary = json.loads((out / "qc_summary.json").read_text())
    assert summary["audit"]["judged"] == len(aud) and summary["audit"]["ok_frac"] == 1.0


def test_wilson_interval():
    lo, hi = qc.wilson(9, 10)
    assert 0.55 < lo < 0.65 and 0.97 < hi <= 1.0

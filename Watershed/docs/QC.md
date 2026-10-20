# Quality control of a run

`python -m nuclei_stage1.qc <volume_dir> --workers 8` (the LSF script runs it automatically after segmentation, and a QC
failure never fails the run). It needs `raw_chunked/` and `labels/` and writes `<volume_dir>/qc/`:

| file | what |
|---|---|
| `qc_report.html` | the **main report**, short and self-contained (images embedded, so it can be emailed): summary, a handful of representative images, counts per check, audit status, limits |
| `qc_detailed.html` | the **detailed report**: every flagged chunk in a filterable table (status, reason, search), plus extra images. Keep it next to `images/` (images are linked, not embedded, so it stays light for thousands of chunks) |
| `qc_chunks.csv` | every chunk: status, reasons, descriptors, nuclei counts and sizes, neighbour counts, exemplar distance |
| `qc_review_list.csv` | every flagged chunk (REVIEW, MISSING and WATCH), most important first |
| `qc_audit_sample.csv` | a random sample of chunks for a person to judge; never overwritten once it exists |
| `qc_summary.json` | headline numbers |
| `images/` | the pictures used in the report |

## The images
Each shows a raw slice, the same slice with nucleus outlines, and a side view. They are chosen so that someone can glance and
feel (or not feel) confident:

* **At least one edge chunk, every time** (up to `--n-edge`, default 3; a corner chunk first). Edges are where tissue ends and
  zero padding begins, so they are where segmentation fails first. "Edge" means a lateral (y or x) face of its ROI; z is not
  counted because an ROI is typically only two chunks deep.
* a few random interior chunks (`--n-interior`) and only the **two** highest-priority flagged chunks (`--n-flagged`, default 2).

The detailed report holds the rest: the next-highest-priority REVIEW chunks (up to `--max-detail-images`, default 150, to keep
the folder small; the table lists all of them regardless) and the densest and sparsest chunks that contain nuclei.

## The flags
Statuses: **REVIEW** (look at it), **WATCH** (one minor reason), **OK**, **PADDING** (mostly zero padding, nothing to segment),
**MISSING** (no label file: the run did not finish or crashed on it).

| reason | weight | meaning |
|---|---|---|
| `missing_output` | 10 | a raw chunk has no label file |
| `unlike_exemplars` | 5 | its image statistics are farther than `--max-distance` (default 0.75) from every ground-truthed exemplar in `exemplars.json`, so the settings used were never validated on anything like it. Same test as `--adaptive` |
| `no_nuclei_but_signal` | 5 | zero nuclei although the raw data has a bright tail (contrast above 3 sigma) |
| `density_jump_vs_neighbors` | 3 | nuclei per megavoxel is more than 3x above or below the median of the neighbouring chunks (needs 3 or more neighbours) |
| `oversized_outlier` | 3 | fraction of nuclei larger than 3x the expected volume is an outlier (robust z > 4) versus the rest of the volume: merged clumps |
| `tiny_outlier` | 2 | fraction smaller than 0.2x expected is an outlier: fragments |
| `size_shift_outlier` | 2 | median nucleus volume is an outlier (log scale, both directions) |
| `ratio_outlier` | 2 | nuclei found / bright blobs detected is an outlier (both directions): the mask is picking up noise or missing nuclei |

REVIEW = priority 5 or more, or two or more reasons. The outlier tests compare each chunk with the rest of *this* volume, so they
need no ground truth, but they cannot see a problem that affects the whole volume.

## The audit (this is the real measurement)
Flags are triage. To learn the actual error rate, open `qc_audit_sample.csv` (a random 1.5% of usable chunks, at least 10,
chosen whether or not they were flagged), look at each chunk in napari or in the images, and write `ok`, `minor` or `major` in the
`verdict` column. Re-run the QC command: the report then states how many were judged ok with a 95% Wilson interval. Keep the
sheet; a re-run never overwrites it. Delete it to draw a new sample.

## Limits
* The flags are heuristics and are **not calibrated against ground truth** (only two corrected chunks exist). A chunk can
  pass every check and still be wrong; a flagged chunk can be fine. Every corrected chunk added to the project (see README,
  "Continuing this work") allows a check of how well flags predict real errors.
* `unlike_exemplars` compares three image statistics only.
* Tiles that are mostly streaks or noise (seen on a lateral edge tile) are flagged `unlike_exemplars` + `no_nuclei_but_signal`; the label reads like missed nuclei, but the right answer for such a tile can be zero nuclei. A person decides. A dedicated streak check is not built yet.
* Image scale uses the configured `voxel_size`, which has not been verified against the microscope metadata.
* `HALO=1` runs write `labels_merged/`, which the report does not read yet.

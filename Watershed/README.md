# nuclei-stage1

Classical, CPU-only nuclei instance segmentation for tiled lightsheet brain volumes:
**foreground mask (per-slice percentile, or an optional band-pass) → distance transform → seeded watershed →
per-object quality screen**, plus stitching for nuclei that straddle tile boundaries, a ground-truth scoring tool, a
tuning grid, an **experimental automatic per-chunk parameter selector**, and viewers for whole volumes.

This repository contains only the part of our cascade that has been checked against human-corrected labels. The
deep-learning refinement stage (StarDist) is **not** included because it has not been validated.

> **Status: a solid, documented starting point — not a finished method.** Read
> [docs/VALIDATION.md](docs/VALIDATION.md) before trusting any number.
> * The defaults were tuned and scored on **one** corrected prefrontal-cortex (PFC) chunk (118 nuclei): F1 0.80,
>   mean matched IoU 0.60.
> * A **second corrected chunk (hippocampus, 74 nuclei)** scored only F1 0.56 with those defaults. A band-pass mask
>   plus retuned seeds (`--preset hippocampus`) brought it to F1 0.80 — tuned and scored on the same chunk, so optimistic.
> * Nothing is validated on cerebellum, and the voxel sizes are unverified.
> * **The most valuable thing you can do is correct more chunks** (see [Continuing this work](#continuing-this-work)).

## Install

```bash
git clone <this repo> && cd nuclei-stage1
pip install -e .              # numpy, scipy, scikit-image, tifffile, pandas
pip install -e ".[view]"      # + napari, only for the viewer
pip install -e ".[test]" && pytest     # ~3 minutes; the slowest tests run a fake volume end to end
```

Python ≥ 3.9. No GPU needed.

## Input layout

```
<volume>/raw_chunked/{roi}_z{Z}_y{Y}_x{X}.tif      one 3-D tile per file (we use 128³), axis order (0,1,2)
```

`(Z, Y, X)` in a name is the tile's offset **inside its ROI file**. Several ROIs of one volume overlap in
coordinates, so everything keys on `(roi, z, y, x)` and ROIs are processed and viewed separately.
Tile offsets must sit on a regular grid (multiples of the tile step); a missing neighbour is treated as empty space.

## Pick a run mode

| You want | Run | Writes | Cost |
|---|---|---|---|
| Fastest; accept cut nuclei at tile edges | `python -m nuclei_stage1.run_slab <volume> --workers 8` | `labels/`, `centroids/` | 1× |
| **Merge nuclei cut by tile edges** | `run_slab` then `python -m nuclei_stage1.stitch <volume> --workers 8` | + `labels_stitched/`, `stitched_centroids.csv` | ≈1× + a cheap merge pass |

Resume after a scheduler time-out with `--skip-existing`. All ROIs of the volume are processed; `stitch` also
accepts `--roi N` to do one.

**Stitching (`stitch`).** Face-adjacent tiles are compared plane to plane; two labels are the same nucleus when
their footprints on the shared plane overlap (≥ 10 px and ≥ 50 % of the smaller footprint). A merge is refused if
it would make an object longer than the length cap, so chains of touching nuclei cannot snowball. It cannot
recover pieces stage 1 already discarded as too small.

**Tissue-specific settings.** Tissue differs (see below), so settings are chosen per volume, not globally:

```bash
python -m nuclei_stage1.run_slab <volume> --workers 8                       # PFC-validated defaults
python -m nuclei_stage1.run_slab <volume> --workers 8 --preset hippocampus   # band-pass mask + retuned seeds
python -m nuclei_stage1.run_slab <volume> --workers 8 --adaptive             # EXPERIMENTAL: settings chosen per tile
```

Every run writes `<volume>/config_choices.csv` recording the settings each tile used.

**Overlapping-edge processing (experimental, not recommended).** `nuclei_stage1/experimental/run_halo.py`
segments each tile inside a block padded with its neighbours' voxels so nuclei are never cut. It works and is
tested, but it was much slower than expected on a full volume and its benefit over stitching is modest, so it is
kept out of the main workflow. Details in [docs/VALIDATION.md](docs/VALIDATION.md).

Boundary-handling numbers from an 8-tile test are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Outputs

* `labels*/…_labels.tif` — `uint32`, 0 = background.
* `centroids/{tile}_centroids.csv` — `label_id, z, y, x, voxel_count`, in **ROI voxel coordinates, native frame**
  (tile offset already added). Multiply by `Stage1Config.voxel_size` for micrometres.
* `timing_per_chunk.csv` and a `TIMING:` line in the log — seconds per tile and effective throughput.
* `qc/qc_report.html` — the short quality-control report (representative images, always including a tile from the edge of
  the volume, plus counts per check), written automatically by the LSF script or by `python -m nuclei_stage1.qc <volume>`.
  `qc/qc_detailed.html` lists every flagged tile with filters; `qc_audit_sample.csv` is the random audit sheet. See
  [docs/QC.md](docs/QC.md).

## Parameters

All in [`nuclei_stage1/config.py`](nuclei_stage1/config.py) (`Stage1Config`); each field is commented with what it
does and what we measured. The ones that matter:

| Parameter | Default | Note |
|---|---|---|
| `voxel_size` | `(2.0, 0.5616, 0.528)` µm | **unverified** — check against your acquisition metadata |
| `target_foreground_percentile` | 90 | should roughly match nucleus density; too high over-splits (94 with the band-pass) |
| `mask_bandpass_um` | `None` (off) | `(lo, hi)` µm difference-of-Gaussians mask, for tissue where nuclei and background brightness overlap (hippocampus) |
| `min_seed_distance` | 17 | ≈ nucleus radius, in voxels of the finest axis |
| `background_floor_sigma` | 1.5 | suppresses noise "nuclei" in empty regions; 0 = off |
| `max_extent_um` | 60 | flags long streaks/columns |
| `seed_landscape_smooth_um` | 0 (off) | 1.0 helped hippocampus, was neutral on PFC; not default |

Override in code with `Stage1Config(target_foreground_percentile=92.0)` or `dataclasses.replace`. Named
tissue-specific sets live in `config.PRESETS` (`Stage1Config(**PRESETS["hippocampus"])`).

## Tuning and scoring

```bash
# score any label volume against corrected labels (Hungarian match at IoU ≥ 0.25)
python -m nuclei_stage1.evaluate --gt gt.tif --z-range 1:127 --pred stage1=labels.tif

# grid-search percentile × seed distance × landscape smoothing × mask type on one corrected chunk
python -m nuclei_stage1.tune --raw gt_raw.tif --gt gt.tif --z-range 1:127 --name pfc --workers 8 \
    --bandpass none,2:8,1:8 --percentiles 90,94 --seed-distances 12,17 --smooth 0,1
# default settings vs a preset on one corrected chunk, with the full side-by-side table
python -m nuclei_stage1.compare_preset --raw gt_raw.tif --gt gt.tif --preset hippocampus
```

Always pass `--z-range` covering only the slices you actually finished correcting; otherwise you are scoring
against uncorrected leftovers. One chunk's best F1 is partly overfitting — `tune` prints how flat the top of the
table is, and `--pick-conservative` takes the fewest-false-positive setting within 0.01 F1.

`experiments/` holds the scripts behind the findings (`sweep_denoise.py`, `sweep_seed_landscape.py`,
`compare_baselines.py`, `label_free_proxies.py`). They need the corrected chunks, which are not part of this repository.

## Viewing a volume

```bash
python -m nuclei_stage1.view_slab <volume>                         # list ROIs and sizes
python -m nuclei_stage1.view_slab <volume> --roi 0 --downsample 4 --labels-dir labels_stitched --deskew
python -m nuclei_stage1.view_slab <volume> --roi all --points      # every ROI, centroids only
# big volumes: build the downsampled mosaic where the data lives, copy it, open locally
python -m nuclei_stage1.view_slab <volume> --roi all --downsample 6 --labels-dir labels_stitched --save-mosaic out/slab
python -m nuclei_stage1.view_slab --open out/slab
```

`--deskew` is display-only (undoes the camera shear). With `--roi all` the ROIs are laid out side by side with a
gap — filenames carry no stage position, so this is **not** their true relative placement. Downsampling is plain
striding, so small nuclei thin out at high strides.

## Saving whole-volume pictures (no display needed)

```bash
python -m nuclei_stage1.save_slab_pair <volume> --roi all --labels-dir labels_stitched
```

Writes `<volume>/figures/<volume>_<roi>_pair.png` per ROI: **raw on the left, the same view with nuclei filled in on
the right**, at the same physical scale. Rows are views (`--views proj_xy,sec_xy,proj_xz,proj_yz`): whole-depth
maximum projections and a thin section. Where nuclei overlap along the line of sight the one with the brightest
voxel is drawn. It uses matplotlib only, so it runs on a cluster node. The count in each title is of nuclei still
visible at the chosen `--downsample`, not the true total.

### Spinning 3-D movie

```bash
python -m nuclei_stage1.spin_slab_pair <volume> --roi all --labels-dir labels_stitched
```

Writes `<volume>/figures/<volume>_<roi>_spin.gif` (plus `.mp4` if `imageio-ffmpeg` is installed) and a still
`_spin_poster.png`: raw volume on the left, nuclei in per-nucleus colours on the right, rotating together. It is a
headless numpy point-splat render (no display, no napari), so it is flatter than napari's isosurface view; use
`view_slab` locally for interactive rotation. If a ROI is too big, raise `--downsample`.

## Continuing this work

What exists is a framework you can extend, not a finished product. In rough order of value:

1. **Correct more chunks.** Take 6–10 small chunks (even ~40 slices each) across regions and depths, correct the
   stage-1 labels by hand (napari works), and score them. This is what everything else is waiting for: it tests whether the
   band-pass mask generalises, and it is the training set for automatic parameter selection.
2. **Register them** so the selector can use them: `python -m nuclei_stage1.tune --raw … --gt … --name <chunk> --bandpass
   none,2:8,1:8 --register`, then `python -m nuclei_stage1.evaluate_dispatch --manifest gt_manifest.json` to see whether
   automatic selection beats applying another chunk's settings blindly. See [docs/AUTO_CONFIG.md](docs/AUTO_CONFIG.md).
3. **Tighten and extend the selector** — it is a nearest-example lookup over three hand-picked image statistics with
   two examples. Ideas (a fitted rule, per-tile confidence, a percentile chosen from the data) are listed there.
4. **Sweep what was not swept after the band-pass**: the instance-screen limits (solidity, volume z-score, min/max
   volume) and `min_object_voxels`.
5. **Verify voxel sizes** against acquisition metadata (`python -m nuclei_stage1.check_voxel_scale <labels>`; the
   third axis looks wrong, which makes renders look flat).
6. Cerebellum has no corrected labels yet. The next step toward the real goal — matching nuclei to cell type through the other
   channels of the registered data — needs a script that measures per-nucleus intensity of each channel inside each label.

7. **Do the QC audit on every volume**: judge the random sample in `qc/qc_audit_sample.csv` and re-run the QC. That, not
   the automatic flags, gives the real accuracy. Add what you find to the exemplars if a region looks unlike the existing ones.

What we learned, including what did *not* work, is in [docs/VALIDATION.md](docs/VALIDATION.md).

## Running on a cluster

[`scripts/submit_slab.lsf`](scripts/submit_slab.lsf) is a generic LSF job (edit the environment line):

```bash
BASE_DIR=/path/to/data SLAB=my_slab_chunks STITCH=1 bsub < scripts/submit_slab.lsf
PRESET=hippocampus BASE_DIR=… SLAB=… bsub < scripts/submit_slab.lsf       # tissue-specific settings
ADAPTIVE=1 BASE_DIR=… SLAB=… bsub < scripts/submit_slab.lsf              # experimental per-tile settings
FAST_EDT=1 BASE_DIR=… SLAB=… bsub < scripts/submit_slab.lsf              # opt-in faster distance transform (pip install edt)
QC=0 BASE_DIR=… SLAB=… bsub < scripts/submit_slab.lsf                   # skip the QC report
```

It moves any earlier `labels*` / `centroids` folders aside with a timestamp instead of overwriting, and refuses to
"succeed" on an empty slab.

## Layout

```
nuclei_stage1/   config (+ PRESETS), segmentation (the algorithm), qc (quality-control report), chunks, run_slab, stitch, evaluate, tune, deskew,
                 descriptors + autoconfig + evaluate_dispatch (experimental auto-selection; exemplars.json = its examples),
                 compare_preset, check_voxel_scale, view_slab, save_slab_pair, spin_slab_pair
  experimental/  run_halo (overlapping-edge processing; works, but slow)
experiments/     sweeps behind the documented findings (need ground truth)
scripts/         LSF job template
docs/            METHODS.md (how it works), VALIDATION.md (evidence, limits, negative results), AUTO_CONFIG.md,
                 QC.md (quality-control report), PERFORMANCE.md (timings, speed-ups)
tests/           synthetic-data tests, including an end-to-end fake volume
```

## Checking voxel size / nucleus shape

Rendered nuclei that look flat or oversized usually mean the configured voxel size is wrong for an axis:

    python -m nuclei_stage1.check_voxel_scale <volume>/labels_merged          # median nucleus extent per axis, in voxels and um
    python -m nuclei_stage1.check_voxel_scale <volume>/labels_merged --voxel-size 2.0,0.5616,0.84

Nuclei are roughly round, so extents (um) should be similar on all three axes. `--voxel-size z,y,x` is also accepted
by `nuclei-view-slab`, `nuclei-save-pairs` and `nuclei-spin`; `nuclei-spin --clip-um 150` draws a thin plate so
individual nuclei and empty space are visible instead of a solid full-depth block.

## Licence and data

No licence has been chosen yet — add one before publishing. No image data or labels are included.

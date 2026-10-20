# Methods

All of this lives in `nuclei_stage1/segmentation.py` and is applied to one array at a time (a tile, or a tile
plus halo).

## 1. Foreground mask (`foreground_mask`)

For every z-slice, keep the voxels brighter than the `target_foreground_percentile`-th percentile of that slice
(default 90 → the brightest 10 %). Per-slice thresholds follow depth-dependent illumination falloff. We use this
instead of Otsu because Otsu marked 29–46 % of a chunk as foreground when nuclei occupy ~3–5 %: the contrast is
too weak for a bimodal histogram. (Otsu modes `global` / `local` remain selectable for comparison.)

A percentile mask *always* keeps the brightest N % of a slice, even a slice that contains no nuclei. The
**background floor** (`background_floor_sigma`, default 1.5) additionally requires a voxel to exceed
`median + k · 1.4826 · MAD` of that slice's non-zero voxels. This mainly removes noise/stripe "foreground" in
empty regions. A strict `>` is used, so perfectly flat plateaus tie with the threshold and drop out — irrelevant
for real data, relevant when you build synthetic tests (add texture).

**Optional band-pass mask (`bandpass_mask`, `mask_bandpass_um=(lo, hi)`).** Where nucleus and background brightness overlap
(our hippocampus chunk), the raw percentile mask is poor. The alternative thresholds G(lo) − G(hi), a difference of Gaussians
with σ in micrometres (so it respects the anisotropic voxel size), with one global percentile: the subtraction removes the
slowly varying background, so no per-slice trend is left to follow. It only decides which voxels are foreground; seeds and
boundaries come from that mask exactly as before. Off by default; see `config.PRESETS` and
[VALIDATION.md](VALIDATION.md).

If a `valid` mask is passed (padding from deskewing or from the edge of a halo block), percentiles use real
voxels only and padding is never foreground.

## 2. Seeds and watershed (`seeded_watershed`)

1. Euclidean distance transform of the mask with `sampling=voxel_size`, so distances are physical.
2. Optional Gaussian smoothing of the distance map (`seed_landscape_smooth_um`); the mask itself is untouched,
   so boundaries still come from the unsmoothed image.
3. Seeds = local maxima of the distance map, using a **physically spherical** exclusion footprint. The radius is
   `min_seed_distance` voxels on the finest axis and is scaled down on coarser axes, so the exclusion region is
   the same physical size in every direction. (A plain `min_distance` is an array-index radius and was
   measurably wrong once real anisotropic voxel sizes were used.)
   Speed: the search is exact but done cheaply — a row-decomposed maximum filter and a sparse candidate test
   (only voxels that could be a maximum are checked against the footprint); it falls back to scikit-image's
   `peak_local_max` for trivial or plateau-heavy blocks. Output is identical (checked on all corrected chunks).
   A smaller `min_seed_distance` means a smaller footprint, which is why the hippocampus preset runs faster.
4. Watershed on `-distance` inside the mask. `fast_edt=True` (opt-in, needs `pip install edt`) computes the
   distance transform in float32 with the `edt` package, ~3× faster; differences ~1e-7, labels were identical on
   our corrected chunks but that is not guaranteed, so it is off by default.
5. Objects smaller than `min_object_voxels` are zeroed **without** re-labelling by connectivity — re-running
   connected components would silently re-merge touching nuclei.

## 3. Instance screen (`explain_instances` / `classify_instances`)

Each object gets a list of reasons it is suspicious; objects with no reasons are *accepted*:

| Reason | Rule | Note |
|---|---|---|
| `solidity` | solidity < `solidity_thresh` (0.15) | very loose: most watershed objects here have low solidity |
| `volume_zscore` | \|z\| > 4 vs the other objects in the block | the block's own population is inflated by big blobs |
| `too_small` | volume < 0.20 × nominal nucleus volume | absolute floor; the z-score alone passes uniformly over-split blocks |
| `too_big` | volume > multiple × nominal | off by default |
| `too_long` | longest bounding-box extent > `max_extent_um` | catches columns/streaks; never auto-accepted |

With `accept_big_side_flags=True`, objects flagged *only* as too big / positive volume outlier are kept: on the
corrected chunk these were almost always whole nuclei drawn generously. `too_small`, `solidity` and `too_long`
are never promoted. `classify_instances` is built on `explain_instances`, so a report and the actual decision
cannot disagree. Solidity is evaluated lazily: an object whose area/bounding-box bound or 2-D projection hull bound already
clears the threshold skips the convex hull (identical decisions, far fewer hulls). Implementation note: it uses lookup tables rather than one full-volume comparison per object,
which matters on large padded blocks.

`run_stage1` returns `(accepted, difficult, labels)`: the pipeline output, a mask of flagged objects, and the
unscreened watershed labels.

## 4. Tile boundaries

* **Stitching (`stitch`, recommended).** Union-find over face-overlap edges, strongest first, refusing merges
  that exceed the length cap. Ids in `labels_stitched/` are global and marked with `.global_ids`.
* **Overlapping edges (`experimental/run_halo.py`, not recommended).** Each tile is segmented inside a block
  padded with `--halo-um` (default 20 µm) of its neighbours' raw voxels; an object belongs to the tile that
  contains its centre of mass and is kept whole. Kept objects are cut into per-target-tile pieces
  (`labels_pieces/`) and each tile is assembled from its own piece plus pieces spilling in from neighbours
  (`labels_merged/`). Ids are `tile_ordinal × 4096 + local`. Objects longer than about twice the halo and centred
  near an edge can still be cut. It was too slow on a full volume to be worth the modest gain.

## 5. Deskew (optional, display-oriented)

The nuclei camera's −7° shear is `X' = X + 0.1228 · Z`. The direction was measured, not assumed: applying the
registration matrix as a forward map doubled the shear in annotated nuclei (residual slope −0.12 → −0.23), while
its inverse removed it (→ +0.02 ± 0.06). `python -m nuclei_stage1.deskew --check labels.tif` repeats that check
on a new corrected chunk. Whether segmenting in the deskewed frame improves accuracy was **inconclusive**, so
`run_slab --deskew` is off by default and the viewer's `--deskew` is display only.

## 6. Scoring (`evaluate`)

Hungarian matching at IoU ≥ 0.25 gives precision / recall / F1 and mean matched IoU. Splits and merges are
counted by *contribution* (a piece accounting for ≥ 15 % of the other side's volume), not by IoU, because an
object cut into three roughly equal pieces has each piece's IoU below any sensible threshold even though all
three clearly belong to it.

## 7. Automatic per-chunk parameters (experimental)

`descriptors.py` + `autoconfig.py`: measure cheap image statistics of a tile, find the nearest registered corrected chunk, and
apply the settings tuned on it (or the defaults if nothing is close). A framework, not a validated method — read
[AUTO_CONFIG.md](AUTO_CONFIG.md) before using it.

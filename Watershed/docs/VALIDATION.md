# Validation: what has been checked, and what has not

## The ground truth

Two corrected chunks, each 128³ voxels, segmented by this pipeline and then **corrected by a human**:

| chunk | tissue | nuclei | scoring window | used for |
|---|---|---|---|---|
| PFC chunk | prefrontal cortex | 118 | z = 1:127 | the default parameters (everything in the sections below unless stated) |
| hippocampus chunk | hippocampus | 74 | whole chunk | the band-pass mask and the `hippocampus` preset (last section) |

Every number that says "validated" means "scored against one of these chunks".

**A note on the PFC labels.** The PFC labels were edited after the first tuning rounds. The current, correct file is
`gt_chunk1.tif`; everything in this document is scored against it unless stated. Against the earlier version of the same chunk the
default settings scored the same F1 (0.796) but mean matched IoU 0.666 instead of 0.602, so the **IoU figures in the 8-tile
table and in the early tuning rows below were measured against the earlier labels and are not directly comparable** to the
current ones. Always score two settings against the same file and say which.

Consequences you should keep in mind:

* The parameters were tuned over several rounds on this chunk (earlier rounds used partial annotations of 45 and
  74 nuclei). Expect some overfitting; the tuning tool reports how flat the optimum is for that reason.
* A different tissue region does need different parameters: the hippocampus chunk scored F1 0.56 with the PFC defaults
  (see the last section). Cerebellum has no corrected labels at all.
* Matching uses IoU ≥ 0.25 (a lenient threshold). Mean *matched* IoU of 0.60 means the shapes that are found are
  only moderately accurate.

## Headline result (defaults, whole chunk processed as one block)

| | predicted | F1 | mean matched IoU |
|---|---|---|---|
| ground truth | 118 | | |
| stage 1, defaults | 93 | 0.796 | 0.60 |

## Tile-boundary handling (8-tile test)

The corrected chunk was cut into 8 sub-tiles (scored against the earlier version of the labels, where the one-block reference had IoU 0.666; F1 is unaffected, IoU is not comparable to 0.60) and run through each approach, then scored against the same labels.

| approach | predicted | F1 | mean matched IoU |
|---|---|---|---|
| whole chunk, one block (reference) | 93 | 0.796 | 0.666 |
| 8 tiles, independent (cut nuclei) | 113 | 0.693 | 0.487 |
| 8 tiles + stitching | 95 | 0.723 | 0.485 |
| 8 tiles, overlapping edges (halo; experimental) | 89 | 0.754 | 0.595 |

Overlapping edges recover most of the loss and, unlike stitching, also recover IoU (0.595 vs 0.485). Stitching
only relabels pieces; our reading is that the halo run also fixes the mask and watershed split near the cut, but
we did not isolate that. It still trails the one-block reference by about
0.04 F1. This is one chunk cut into small tiles, a harsher test than 128³ production tiles; it has not been
repeated on real tile seams.

## Things that worked

| Finding | Evidence |
|---|---|
| Percentile mask instead of Otsu | Otsu: 29–46 % foreground vs 3–5 % real signal |
| Foreground percentile 90 instead of 96 | 96 lost ~20 % of nuclei before any later stage could see them (measured against a pixel-classifier proxy) |
| Physically spherical seed footprint | Index-radius seeding gave more merged objects / lower recall once real anisotropic voxel sizes were used |
| `min_seed_distance` 17 | sweep 4–30 on the 74-nucleus annotation; 17–19 best, 17 chosen because it lost no matches. A precision/recall trade, not a strict win |
| `seed_prominence` 0.15 | 0.10–0.15 tie; **inert** at the final settings |
| `accept_big_side_flags` | oversized-only flags were almost always whole nuclei; keeping them raised matches with no geometry change |
| `min_volume_fraction` 0.20 | clear gain over 0.06–0.10 on the 45-nucleus annotation (precision up, recall unchanged) |
| Seed-landscape smoothing (**optional**) | 1.0–1.5 µm Gaussian on the distance map with `min_seed_distance` 10–12: F1 0.83–0.86, IoU 0.62–0.63 vs 0.796 / 0.602 (rescored on the current labels). Helped hippocampus too, but the gain on PFC is from one chunk, so not default |
| Lookup-table instance screen | identical output to the per-object version; ~2× faster on a synthetic padded block |

## Things that did not work (kept so nobody repeats them)

* **Mask smoothing** (opening ± closing, radius 1–2): worse on every metric except the split count; recall fell
  from 0.82 to 0.31 at radius 2. Left off.
* **Whole-image denoising before segmentation** (Gaussian, median, background subtraction): lowered IoU by
  blurring boundaries. Smoothing only the seed landscape, never the mask, is what helped.
* **Solidity as a discriminator**: nearly every watershed object here has low solidity, so it cannot separate
  blobs from nuclei. The threshold is set low enough to be nearly inert.
* **Absolute volume ceiling** (2.5× nominal): removed 3 of 6 oversized objects but cost 2 of 45 true matches.
  Calibrated on ground-truth volumes but applied to predicted volumes, which run ~1.4× larger. Left off.
* **"Mean intensity < background + 3σ" as a false-positive test**: invalid. 63 % of verified real nuclei fail it;
  real nuclei sit only ~2.8σ above background. Do not use a brightness-versus-background rule to call objects
  false positives.

## Added but **not** validated against labels

* **Background floor (`background_floor_sigma` 1.5).** On the corrected chunk, k ≤ 1.5 leaves the result
  unchanged (F1 0.796) and k = 2 costs F1 (0.761). On a sparse chunk with no labels it removed most
  background-level objects (65 → 17 of ~74), but without labels we cannot say they were all false.
* **Length cap (`max_extent_um` 60).** Added after seeing long, thin objects in an empty region of a slab. It is
  a sanity bound on physical size, not a tuned parameter, and depends on `voxel_size` being right.
* **Overlapping-edge run time.** The padded block has ~2.9× the voxels of a tile (default 20 µm halo, default
  voxel size). On a real volume the run took much longer than we expected, so it was moved to `experimental/` and
  is not the recommended path. We did not record a full-slab timing; if you revisit it, profile first (the
  per-object screening was already made ~2× faster), try a smaller halo (e.g. 12 µm), and compare against
  stitching on the same tiles.
* **Deskewing before segmentation.** The shear direction was measured on 113 annotated nuclei, but whether
  segmenting in the deskewed frame improves F1 was inconclusive.

## Known unknowns

* **Voxel size.** `(2.0, 0.5616, 0.528)` µm comes from a camera-spec note, not acquisition metadata. The extents
  of the corrected nuclei suggest roughly (2.2, 0.56, 0.84) µm for round nuclei, so the third value in
  particular deserves checking. Voxel size affects the distance transform, the seed footprint, the halo size and
  the length cap.
* **Transfer to other regions.** Untested (see above).
* **Per-region parameter selection.** An automatic scheme that picks the mask type, percentile and seed distance from image
  statistics is included as an experimental framework (`autoconfig.py`, [AUTO_CONFIG.md](AUTO_CONFIG.md)). It is a
  nearest-example lookup over two corrected chunks and cannot yet be said to generalise.

## What would make the claims stronger

1. **More corrected chunks, 6–10 across regions and depths** (even ~40 slices each) — the single most useful thing: it is what
   automatic parameter selection needs, and what would show whether the band-pass mask generalises.
2. A held-out corrected chunk from the same region that was not used for tuning (for both PFC and hippocampus).
3. A second look at `seed_landscape_smooth_um = 1.0` on more chunks, then promote it to the default if it holds.
4. Verified voxel sizes.
5. A scored comparison of stitching versus the experimental halo run on real production seams, with timings.

## Hippocampus chunk and the band-pass mask

A second corrected chunk (hippocampus, 74 nuclei) scored **F1 0.558, mean matched IoU 0.395** with the PFC defaults — much
worse than PFC. Sweeping percentile, seed distance and smoothing alone only reached F1 0.60 and IoU stayed near 0.40, which
pointed at the foreground mask rather than the seeds.

**Why.** Nuclei fill only 6 % of this chunk and their brightness overlaps the background (median 4192 inside labels vs 3376
outside; the median inside labels barely exceeds the 90th percentile outside, 3968). The best Dice any raw per-slice percentile mask could
reach against the corrected labels was 0.43 (global percentile 0.46).

**Fix tried: a band-pass mask.** Threshold G(σ_lo) − G(σ_hi) (difference of Gaussians, σ in µm) with one global percentile.
Mask Dice against the corrected labels: raw 0.43 → Gaussian 1 µm 0.55 → DoG 1–6 µm 0.59 → DoG 1.5–6 µm 0.63 → DoG 2–8 µm
**0.65**. Everything downstream (seeds, watershed, screen) is unchanged.

Full stage 1 on the hippocampus chunk (IoU ≥ 0.25 matching), PFC defaults vs the tuned preset
(`hippocampus`: band-pass 2–8 µm, percentile 94, `min_seed_distance` 12, seed smoothing 1 µm):

| | default | `hippocampus` preset |
|---|---|---|
| predicted / GT | 73 / 74 | 76 / 74 |
| matched | 41 | 60 |
| missed | 33 | 14 |
| extra (false positives) | 32 | 16 |
| GT nuclei split | 7 | 5 |
| predictions merging nuclei | 3 | 1 |
| precision / recall | 0.562 / 0.554 | 0.789 / 0.811 |
| **F1** | **0.558** | **0.800** |
| mean matched IoU | 0.395 | 0.521 |

**Caveats.**
* Tuned and scored on the same 74 nuclei, so optimistic. The top of the search is flat: about a dozen neighbouring settings are
  within ~0.03 F1, so the exact numbers are soft.
* **Not for PFC.** Applied to the PFC chunk the preset raises F1 slightly (0.796 → 0.816) but drops mean matched IoU from 0.602 to
  0.524, i.e. worse boundaries. A band-pass tuned for PFC itself (1–8 µm, percentile 94, seed distance 17) reached F1 0.834 and
  IoU 0.564, again higher F1 but lower IoU than the default; we did not adopt it. Simple seed smoothing alone (1.0 µm,
  seed distance 12) reached F1 0.828 and IoU 0.626 on PFC, better on both, which is the more promising PFC change. This is why
  presets are per tissue, and why PFC was left at its defaults.
* Residual errors after the preset: 14 missed, 16 extra, 5 split, 1 merged; matched IoU 0.52 means outlines are still loose.
* Seed smoothing 1 µm helped the hippocampus chunk (F1 0.711 → 0.752 at seed distance 17, band-pass 1–8 µm, percentile 94) and was
  neutral to slightly negative on the PFC chunk when combined with the band-pass (0.834 → 0.825); seed prominence (0.10–0.25) did nothing; seed distance 12 vs 14 vs 17 was within noise on F1.
* Not swept after the band-pass: the instance-screen limits (solidity, volume z-score, min/max volume) and `min_object_voxels`.
* The band-pass percentile is global per tile, so an empty tile still receives ~(100 − percentile) % foreground.

## Label-free quality scores do not pick the settings

We tested whether the segmentation's own statistics could choose settings without labels (fraction of objects or voxels passing
the screen, spread of accepted sizes, accepted count, mask fraction). Rank correlations with F1 were +0.3 to +0.9 but none selected
the best setting, and the screen-pass score would have chosen a PFC setting (F1 0.775) below the validated default (0.796). Details
in [AUTO_CONFIG.md](AUTO_CONFIG.md); script in `experiments/label_free_proxies.py`.

## Rendering and voxel-size checks (visual pitfalls we hit)

* **Flat / elongated nuclei in a render** usually mean a wrong voxel size on one axis. `python -m nuclei_stage1.check_voxel_scale
  <labels>` prints median nucleus extents per axis in µm; round nuclei should give similar values. The extents of the corrected PFC
  nuclei imply roughly (2.2, 0.56, 0.84) µm, so the third axis (configured 0.528) deserves checking against acquisition metadata.
* **"Nuclei fill everything, no black space"** in the whole-volume pair/spin renders had three causes: the raw panel draws only the
  brightest few percent of voxels while the label panel draws all labelled voxels; a whole ROI is millimetres deep, so a full-depth
  view piles up (use `nuclei-spin --clip-um 150` for a thin plate); and predicted volumes run ~1.4× larger than corrected ones.
* **"0 nuclei" renders** came from pointing at a labels folder that did not exist or was empty (e.g. `labels_merged` vs `labels`).
  `view_slab`/`save_slab_pair`/`spin_slab_pair` now exit with a list of the labels folders they found rather than drawing nothing.

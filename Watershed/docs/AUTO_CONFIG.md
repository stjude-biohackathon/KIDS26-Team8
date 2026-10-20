# Automatic per-chunk parameter selection

**Status: a working framework that nobody should rely on yet.** It is meant as an example for others to extend. It has been
exercised on exactly two corrected chunks, so any number it produces is an anecdote until there are more. The pipeline's
default behaviour is unchanged: nothing here runs unless you pass `--adaptive` (or `ADAPTIVE=1` to the LSF script).

## Why this exists
Tissue differs, and a single set of parameters does not suit all of it. On the hippocampus chunk (weak contrast) the plain
per-slice percentile mask tops out at Dice 0.43 against the corrected labels, and a band-pass mask reaches 0.65 and lifts
F1 from 0.56 to 0.80. On PFC the validated defaults are the right choice. Labelling and sweeping every chunk of a
petabyte-scale dataset is impossible, so the question is whether the computer can find the right setting itself.

## How it works
1. `descriptors.py` measures cheap image statistics per chunk: foreground fraction above the chunk's own background,
   blob density, bright-tail contrast (~0.3 s per 128³ chunk, ~3 s per 256³).
2. `autoconfig.py` compares them (log space) with the **registered examples** in `nuclei_stage1/exemplars.json` and applies
   the nearest example's settings: mask type (`mask_bandpass_um`), percentile, seed distance, seed smoothing, and so on.
3. If no example is within `--max-distance`, the chunk gets the global defaults and says so.
4. Every choice is written to `<volume>/config_choices.csv` (source, nearest example, distance, mask, percentile, descriptors).

Nothing is fitted and no labels are needed at run time. But the examples come from corrected chunks, so the scheme can only
recognise tissue that resembles something you have already corrected.

## Use
```bash
python -m nuclei_stage1.run_slab <volume> --adaptive --workers 8      # automatic per-tile settings
python -m nuclei_stage1.autoconfig survey <volume> --n 40             # dry run: which setting would each sampled tile get?
python -m nuclei_stage1.autoconfig select tile.tif                  # what would ONE tile get?
```
Tissue-specific alternative with no automation: `--preset hippocampus` (see `config.PRESETS`).

## Adding a corrected chunk (this is how it improves)
```bash
python -m nuclei_stage1.tune --raw gtN_raw.tif --gt gtN.tif --z-range 1:127 --name hippo_chunkN \
    --bandpass none,2:8,1:8 --percentiles 90,92,94,96 --seed-distances 12,17 --smooth 0,1 --workers 4 --register
python -m nuclei_stage1.evaluate_dispatch --manifest gt_manifest.json     # leave-one-chunk-out check of the whole scheme
```
`--register` stores that chunk's statistics plus the best setting. `evaluate_dispatch` holds each chunk out in turn and
reports, per chunk, the default F1, the F1 from the automatic choice ("dispatch"), a control that applies another chunk's
settings blindly ("shuffle"), and the best possible ("oracle"). Dispatch must beat shuffle for the descriptors to be doing
real work. Manifest format: a JSON list of `{"name", "raw", "gt", "z_range"}`.

## Currently registered
| example | settings | tuned F1 |
|---|---|---|
| `pfc_chunk1` | validated defaults (no overrides) | 0.796 |
| `hippo_chunk3` | band-pass 2–8 µm, percentile 94, seed distance 12, seed smoothing 1 µm | 0.800 |

## What we measured
**Leave-one-out, 2 chunks (a smoke test, not evidence).** Grid: mask none / 2:8 / 1:8, percentile 90 / 94, seed 12 / 17.

| held-out | default | dispatch | shuffle | oracle |
|---|---|---|---|---|
| PFC chunk | 0.796 | 0.816 | 0.816 | 0.847 |
| hippocampus chunk | 0.558 | 0.730 | 0.730 | 0.800 |

With two chunks each held-out chunk has a single example, so dispatch equals shuffle by construction and the test cannot say
whether the descriptors help. The PFC "oracle" is a band-pass setting; the registered `pfc_chunk1` example deliberately keeps
the validated defaults, because applying the band-pass mask to PFC lowered boundary accuracy (see VALIDATION.md).

**Label-free quality scores were tested and are not good enough to choose settings.** Over 16 settings on two chunks:

| score | rank correlation with F1 (hippocampus / PFC) |
|---|---|
| fraction of detected objects passing the size/shape screen | +0.47 / +0.36 |
| fraction of object voxels passing the screen | +0.77 / +0.33 |
| number of accepted nuclei | +0.77 / +0.90 |
| spread of accepted sizes (lower is better) | −0.42 / −0.53 |

They trend the right way but cannot separate near-ties, and letting the screen-pass score decide would have put PFC below its
validated F1. Reproduce with `experiments/label_free_proxies.py`.

**Cost.** Selection takes about 0.3 s per 128³ tile (about 3 s per 256³). Since the segmentation itself was sped up to ~1 s per tile in our container, this is now roughly 20–35 % of per-tile time, so `--adaptive` is not free any more.

**Check on the four ground-truth chunks (all with the two registered examples).** Chunks 1 (PFC) and 3 (hippocampus) match their own examples at distance 0, which is circular. Chunk 4 (hippocampus, not registered) is matched to the hippocampus example (distance 0.27 vs 0.51 to PFC), which is correct. Chunk 2 (PFC with streaks along an edge) is 1.14 / 1.00 from the examples, so the guard rejects it and it gets the defaults; streaks inflate the blob-density descriptor, so descriptors cannot tell them from nuclei. The band-pass mask adds two Gaussian blurs.

## Known limits
* **Two examples.** It can only tell "looks like PFC chunk1" from "looks like hippo chunk3". Aim for several corrected chunks
  per distinct tissue/density regime (even ~40 slices each), across regions and depths.
* **Examples are tuned and scored on the same chunk**, so their F1s are optimistic.
* **The match guard is loose.** The two examples are only ~0.73 apart in descriptor space and the default guard is 0.75, so it
  rejects almost nothing. Tighten `--max-distance` (try 0.4–0.5) once there are more examples.
* **Three hand-picked descriptors**, not tested for how well they predict which mask works. A chunk of mostly empty tissue or a
  different stain could look like either example.
* The band-pass percentile is global per tile, so an empty tile still gets ~(100 − percentile) % foreground.
* Only mask type and percentile mattered much on the chunks tested; seed distance, prominence and smoothing were flat to about ±0.03 F1.

## Ideas, not built
* Fit a small rule or regression from descriptors to (mask type, percentile) once ~10 calibration chunks exist.
* A per-tile confidence from agreement between nearby examples; flag low-confidence tiles for review.
* Choose the percentile per tile from the band-pass image's own histogram.
* Learn which descriptors predict the right mask (the current three were chosen by hand).

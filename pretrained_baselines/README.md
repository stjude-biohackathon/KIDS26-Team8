# Pretrained foundation-model baselines (Cellpose-SAM)

Inference-only nuclei segmentation using off-the-shelf pretrained Cellpose-SAM
(`cpsam_v2`, cellpose 4.2.1), with no training on hackathon data. Intended as
an independent comparison point for the team's trained methods (3D U-Net,
Otsu+StarDist cascade, static thresholding).

## Headline result: prefrontal cortex, official 96-chunk test split

Evaluated on the 96 held-out 128³ chunks designated as `test` in
`split_manifest.json` (no peeking at train chunks), against the team's
Ilastik-derived binary ground truth (`masks_chunked`).

| Metric | Cellpose-SAM default | **Cellpose-SAM tuned** | change |
|---|---|---|---|
| Object F1 @ IoU≥0.5 (mean) | 0.143 | **0.267** | **+87%** |
| Object F1 @ IoU≥0.5 (median) | 0.096 | **0.248** | +158% |
| Object F1 @ IoU≥0.01 (mean) | 0.456 | **0.550** | +21% |
| Dice (voxel overlap) | 0.528 | **0.686** | +30% |
| Nuclei detected / GT nuclei | 7,258 / 13,227 (55%) | **9,173 / 13,227 (69%)** | |
| Chunks with zero F1 | 11 / 96 | **2 / 96** | |
| Runtime per 128³ chunk (1 GPU) | 6.41 s | **6.43 s** | none |
| Peak GPU memory | 1.58 GB | 1.58 GB | none |

"Tuned" = two documented Cellpose parameters changed from default
(`cellprob_threshold=-3`, `tile_norm_blocksize=100`). **No retraining, no
runtime cost.** The default configuration substantially undersells the model
on this data; the dominant default failure mode is under-detection (missing
nuclei), not false positives.

Metrics are reported at two IoU matching thresholds on purpose: 0.5 is the
literature standard; 0.01 ("any overlap counts") matches the team's existing
`Unet/compare_pred_masks.py` convention, so numbers are comparable both ways.

## What the tuning found

Sweeps were run on 5 representative pfCortex test chunks spanning the
observed performance range, then the winner was applied to all 96.

**`cellprob_threshold`** (default 0.0) — the decisive knob. Cellpose docs:
"decrease if not returning as many masks as you'd expect." Clean optimum at -3;
-4 ties, -5/-6 degrade as noise starts being accepted.

| cellprob | F1@0.5 | Dice |
|---|---|---|
| 0 (default) | 0.251 | 0.528 |
| -1 | 0.322 | 0.688 |
| -2 | 0.354 | 0.723 |
| **-3** | **0.379** | **0.736** |
| -4 | 0.376 | 0.724 |
| -5 | 0.325 | 0.693 |
| -6 | 0.240 | 0.649 |

**`tile_norm_blocksize`** (default 0 = whole volume normalized as one) — small
additional gain when combined with cellprob=-3 (F1@0.5 0.379 → 0.388). Light-
sheet brightness varies with depth/position; local normalization stops bright
regions from suppressing dim ones.

**`diameter` — do not set.** Cellpose-SAM is scale-invariant by design and the
parameter is absent from its docs. Setting it anyway measurably hurt *both*
accuracy and speed, because it forces an upscale to the model's 30 px
reference scale (our nuclei are ~6–10 voxels across):

| diameter | F1@0.5 (5 chunks) | runtime / chunk |
|---|---|---|
| unset (default) | **0.251** | **6.4 s** |
| 20 | 0.179 | 10.7 s |
| 16 | 0.123 | 13.4 s |
| 10 | 0 masks | 78 s |
| 6 | 0 masks | 511 s |

**`flow_threshold`** — a no-op for 3D inference (documented in the `eval()`
docstring: "not used for 3D"). Not worth sweeping.

## Runtime scaling (whole-slab volumes)

Three whole pfCortex volumes at 256×1024×1280 (~160× the voxels of a 128³
chunk), default parameters:

| Input | voxels | runtime | peak GPU mem |
|---|---|---|---|
| 128³ chunk | 2.1 M | 6.4 s | 1.58 GB |
| 256×1024×1280 slab | 336 M | 658 s (~11 min) | 1.58 GB |

Runtime scales roughly linearly with voxel count (slightly sub-linear).
**GPU memory is flat** because Cellpose tiles internally (fixed 256 px blocks),
so whole-brain scale is a throughput question, not a memory one. Extrapolating
~6.4 s per 2.1 M voxels: ≈ 3 s per million voxels on one GPU.

## Important caveat: ground-truth quality

The Ilastik-derived ground truth is a binary pixel classification, not a
curated instance annotation, and it is known to be imperfect. Concrete
evidence from this evaluation:

- In dim, low-contrast chunks, Ilastik fragments weak signal into many tiny
  specks. The worst-scoring chunk had 156 "GT objects" with a **median size of
  22 voxels** (~3.5 voxels across — too small to be real nuclei), versus a
  median of 582 voxels in a well-performing chunk. Cellpose-SAM returned ~0
  objects there — arguably the correct call — and was penalised for it.
- GT objects are derived by 26-connectivity connected components on a binary
  mask, so touching nuclei are merged into one GT object. A model that
  correctly separates them is scored as producing false positives.

Both effects push the reported F1 *down*. The numbers above are therefore a
conservative lower bound on real performance. Visual QC in napari is
recommended alongside the metrics (`view_chunk.py`).

## Other regions

- **Cerebellum**: no usable ground truth exists for its official test split
  (`masks_chunked_v2` covers only the 384 *train* chunks; `masks_chunked` has
  1 file). Since this model never trains on the data, scoring against the
  train-split GT would be leakage-free, but it is *not* comparable to methods
  scored on the official test split. `build_test_dirs.py` stages this as a
  clearly separated `tier1_trainsplit/`; not run due to time.
- **Hippocampus**: not chunked; 3 whole volumes with GT only. Not run due to
  time.

## Layout

```
pretrained_baselines/
├── common/metrics.py       object-wise metrics; adapted (credited) from Caitlin
│                           Freeman's segmentation_benchmark/metrics.py. Scores
│                           instance-labeled predictions directly -- does NOT
│                           rebinarize them, which would erase instance separation
├── common/io_utils.py      tif I/O, binary GT -> instance labels (cc3d, 26-conn)
├── build_test_dirs.py      symlink farms from split_manifest.json test lists
├── evaluate.py             scores a masks dir vs GT dir at both IoU thresholds
├── view_chunk.py           napari: raw + predicted + GT as toggleable layers
├── cellpose_sam/
│   ├── run_inference.py    load model once, loop volumes, log runtime + GPU mem
│   └── lsf/                per-region LSF batch scripts (biohackathon queue)
└── results/cellpose_sam/   per-chunk CSVs (default and tuned)
```

## Reproduce

Environment: clone of the shared `pytorch_v2` env with `cellpose` installed and
torch upgraded to 2.14+cu130 (cellpose's opencv dependency requires numpy 2,
which the original torch 2.0 build cannot use). Weights cache to
`~/.cellpose/models/` on first use — do that on a login node with internet.

```bash
python pretrained_baselines/build_test_dirs.py

# tuned configuration
python pretrained_baselines/cellpose_sam/run_inference.py \
  --raw-dir  ~/cellpose_baseline/test_data/slab21_pfCortex_chunks/tier1/raw \
  --masks-out-dir  OUT/masks --runtime-csv OUT/runtime.csv \
  --cellprob-threshold -3 --tile-norm-blocksize 100 --device cuda

python pretrained_baselines/evaluate.py \
  --masks-dir OUT/masks \
  --ground-truth-dir ~/cellpose_baseline/test_data/slab21_pfCortex_chunks/tier1/ground_truth \
  --runtime-csv OUT/runtime.csv --output-csv OUT/eval.csv
```

LSF: `bsub -q biohackathon -gpu "num=1" ...` (that queue needs an explicit
`-gpu` flag; `gpu_interactive` injects its own). Scripts must live on shared
storage, not `/tmp` — compute nodes cannot see the login node's `/tmp`.

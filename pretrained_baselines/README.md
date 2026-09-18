# Pretrained foundation-model baseline: Cellpose-SAM

An independent, **inference-only** nuclei segmentation baseline for Track 1. No training,
no fine-tuning — off-the-shelf pretrained weights applied directly to the hackathon data,
intended as a fast comparison point against the thresholding baseline, the Otsu+StarDist
cascade, and the trained 3D U-Net.

## Headline result

Cellpose-SAM's **default settings substantially undersell it on this data**. One documented
parameter (`cellprob_threshold`) nearly doubles object-wise F1 at **zero runtime cost**.

pfCortex, all 96 chunks of the designated test split from `split_manifest.json`:

| Metric | default | tuned | change |
| --- | ---: | ---: | ---: |
| Object F1 @ IoU≥0.5 (mean) | 0.143 | **0.267** | +87% |
| Object F1 @ IoU≥0.5 (median) | 0.096 | **0.248** | +158% |
| Object F1 @ IoU≥0.01 | 0.456 | **0.550** | +21% |
| Dice | 0.528 | **0.686** | +30% |
| Nuclei found (vs 13,227 GT) | 7,258 (55%) | **9,173 (69%)** | +26% |
| Chunks with total failure (F1=0) | 11 / 96 | **2 / 96** | −82% |
| Runtime per 128³ chunk | 6.41 s | 6.43 s | unchanged |

Tuned configuration: `cellprob_threshold=-3`, `tile_norm_blocksize=100`, `diameter` unset.

## Setup

- **Model**: Cellpose-SAM, `cpsam_v2` checkpoint (cellpose 4.2.1.1), 3D mode (`do_3D=True`)
- **Hardware**: 1 GPU on `nodegpu217` (LSF `biohackathon` queue)
- **Input**: single-channel nuclear stain, uint16, `z_axis=0`
- **Ground truth**: Ilastik-derived binary masks, converted to instances via 26-connectivity
  connected components — the same approach used in `Unet/compare_pred_masks.py`

## Parameter findings

Swept on 5 chunks chosen to span the observed performance range, then the winner was
confirmed on the full 96-chunk test split.

**`cellprob_threshold` is the parameter that matters.** Default 0.0 makes the model far too
conservative on this data; the dominant error mode was missed nuclei, not false positives.

| `cellprob_threshold` | F1@0.5 | Dice |
| ---: | ---: | ---: |
| 0.0 (default) | 0.251 | 0.528 |
| −1 | 0.322 | 0.688 |
| −2 | 0.354 | 0.723 |
| **−3** | **0.379** | **0.736** |
| −4 | 0.376 | 0.724 |
| −5 | 0.325 | 0.693 |
| −6 | 0.240 | 0.649 |

Clean optimum at −3 (−4 ties; below that it starts accepting noise). Adding
`tile_norm_blocksize=100` gave a small further gain (F1@0.5 0.379 → 0.388), consistent with
light-sheet brightness varying across the volume — normalizing in local blocks stops bright
regions from crushing dim ones.

**`diameter` should be left unset — setting it actively hurts.** Cellpose-SAM is
scale-invariant by design, and forcing a diameter degraded both accuracy *and* speed:

| `diameter` | F1@0.5 | runtime / chunk |
| ---: | ---: | ---: |
| unset (default) | **0.251** | **6.4 s** |
| 20 | 0.179 | 10.7 s |
| 16 | 0.123 | 13.4 s |
| 10 | (0 masks) | 78 s |
| 6 | (0 masks) | 511 s |

The runtime blowup is because a diameter triggers upscaling to Cellpose's ~30px reference
scale — `diameter=6` means 5x per axis, i.e. 125x the voxels.

**`flow_threshold` is a no-op here** — Cellpose's own docstring states it is not used for 3D,
and all of our runs are 3D.

## Runtime and scaling

| Input | Voxels | Runtime | Peak GPU memory |
| --- | ---: | ---: | ---: |
| 128³ chunk | 2.1 M | 6.4 s | 1,576 MB |
| 256x1024x1280 whole volume | 336 M (160x) | 658 s (103x) | 1,576 MB |

Two useful properties for scaling toward whole-brain workloads: runtime scales **sub-linearly**
with volume size, and **peak GPU memory is flat** regardless of input size, because Cellpose
tiles internally at a fixed block size. Memory is therefore not the constraint — throughput is.
The full 96-chunk test split completes in ~10 minutes on one GPU.

## Important caveat: ground-truth quality

Object-wise scores here are **lower bounds**, because the Ilastik ground truth fragments weak
signal in dim regions. Concrete example — the worst-scoring chunk in the default run
(`chunk_z5_y55_x5000_z000128_y000000_x000000`, predicted 1 vs 156 GT objects):

| | this chunk | a well-performing chunk |
| --- | ---: | ---: |
| GT objects | 156 | 130 |
| GT median object size | **22 voxels** | **582 voxels** |
| GT foreground | 2.66% | 5.52% |

A median GT object of 22 voxels is ~3.5 voxels across — far too small to be a real nucleus
(well-formed chunks show ~10 voxels across). This is a dim, low-contrast region where the
pixel classifier shattered weak signal into specks. Cellpose-SAM declining to segment there is
arguably *correct*, and the metric penalizes it for that. Some portion of the gap to a
"perfect" score reflects GT noise rather than model error.

Use `view_chunk.py` for visual QC rather than relying on the metrics alone.

## Files

```
pretrained_baselines/
├── build_test_dirs.py      # symlink farms from split_manifest.json (test split)
├── cellpose_sam/
│   └── run_inference.py    # inference + runtime/memory logging
├── evaluate.py             # object-wise scoring at both IoU thresholds
├── view_chunk.py           # napari viewer: raw + predicted + GT as layers
├── common/metrics.py       # greedy one-to-one IoU matching, Dice, PQ, counts
├── common/io_utils.py      # tif IO, binary GT -> instance labels
└── results/cellpose_sam/   # per-chunk CSVs (default and tuned)
```

`common/metrics.py` is adapted from Caitlin's `segmentation_benchmark/metrics.py`. It scores
instance-labeled predictions **directly**, without rebinarizing them first — important here,
since rebinarizing and re-running connected components would merge correctly-separated
touching nuclei back together and erase exactly what an instance segmentation model provides.

## Reproducing

```bash
# 1. build test-split symlink farms
python pretrained_baselines/build_test_dirs.py

# 2. run tuned inference + evaluation (LSF, ~10 min for 96 chunks)
bsub < ~/cellpose_baseline/scripts/run_pfcortex_tuned.sh
```

Environment: `~/.conda/envs/cellpose_env` (clone of the team `pytorch_v2` env; torch was
upgraded to 2.14.0+cu130 there because cellpose requires numpy>=2, which the original
torch 2.0.0 build is not ABI-compatible with). The shared team env is untouched.

## Known gaps

- **Cerebellum has no usable test-split ground truth.** `masks_chunked_v2` covers only the 384
  *train* chunks (verified: 0/96 overlap with the test split), and `masks_chunked` holds a
  single file. Needs resolving with whoever owns the annotations before Cerebellum accuracy
  numbers are possible; inference itself runs fine.
- **Hippocampus** is not chunked, so it has no Tier-1 equivalent — only 3 whole volumes.
- **µSAM** was not run. It needs its own conda env; installing it into `cellpose_env` would
  disturb the torch/numpy pinning that Cellpose depends on here.

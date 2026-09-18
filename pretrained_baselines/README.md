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

Tuned configuration: `cellprob_threshold=-3`, `tile_norm_blocksize=100`, `diameter` left unset.

## Viewing results in napari

All predicted masks are instance-labeled TIFFs (each nucleus a unique integer, 0 = background).
They are too large for git and live on the HPC home directory, not in this repo.

`view_chunk.py` opens the raw volume plus any masks as toggleable napari layers. 

### Where everything is

| What | Path |
| --- | --- |
| Raw 128³ chunks | `/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/raw_chunked/<chunk>.tif` |
| Ilastik ground truth (binary) | `/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/masks_chunked/<chunk>.tif` |
| **Tuned** Cellpose-SAM masks | `/home/efoste34/cellpose_baseline/runs/cellpose_sam_tuned/slab21_pfCortex_chunks/tier1/masks/<chunk>.tif` |
| Default Cellpose-SAM masks | `/home/efoste34/cellpose_baseline/runs/cellpose_sam/slab21_pfCortex_chunks/tier1/masks/<chunk>.tif` |
| Raw whole volumes (256×1024×1280) | `/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/raw/<volume>.tif` |
| Whole-volume ground truth | `/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/masks/<volume>.tif` |
| Whole-volume Cellpose-SAM masks (default) | `/home/efoste34/cellpose_baseline/runs/cellpose_sam/slab21_pfCortex_chunks/tier2/masks/<volume>.tif` |

The 96 test-split chunk names are the filenames in any of the `tier1/masks/` directories
above, or the `chunk` column of the CSVs in `results/cellpose_sam/`. The three whole volumes
are `chunk_z5_y36_x0.tif`, `chunk_z5_y45_x2500.tif`, `chunk_z5_y55_x5000.tif`.

### relevant file paths

Tuned vs. ground truth on a well-performing chunk (tuned F1@0.5 = 0.56):

raw: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/raw_chunked/chunk_z5_y55_x5000_z000000_y000768_x001024.tif
predicted: /home/efoste34/cellpose_baseline/runs/cellpose_sam_tuned/slab21_pfCortex_chunks/tier1/masks/chunk_z5_y55_x5000_z000000_y000768_x001024.tif 
ground-truth: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/masks_chunked/chunk_z5_y55_x5000_z000000_y000768_x001024.tif

The dim chunk where ground truth is noisy

raw: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/raw_chunked/chunk_z5_y55_x5000_z000128_y000000_x000000.tif
predicted: /home/efoste34/cellpose_baseline/runs/cellpose_sam_tuned/slab21_pfCortex_chunks/tier1/masks/chunk_z5_y55_x5000_z000128_y000000_x000000.tif
ground-truth: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/masks_chunked/chunk_z5_y55_x5000_z000128_y000000_x000000.tif

A dense whole volume (predicted 9,864 vs 16,350 GT nuclei at default settings):

raw: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/raw/chunk_z5_y55_x5000.tif
predicted: /home/efoste34/cellpose_baseline/runs/cellpose_sam/slab21_pfCortex_chunks/tier2/masks/chunk_z5_y55_x5000.tif
ground-truth: /lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab21_pfCortex_chunks/masks/chunk_z5_y55_x5000.tif

## Setup

- **Model**: Cellpose-SAM, `cpsam_v2` checkpoint (cellpose 4.2.1.1), 3D mode (`do_3D=True`)
- **Hardware**: 1 GPU on `nodegpu217` (LSF `biohackathon` queue)
- **Input**: single-channel nuclear stain, uint16, `z_axis=0`
- **Ground truth**: Ilastik-derived binary masks, converted to instances via 26-connectivity
  connected components — the same approach used in `Unet/compare_pred_masks.py`

## Parameter tuning

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

## Runtime and scaling

| Input | Voxels | Runtime | Peak GPU memory |
| --- | ---: | ---: | ---: |
| 128³ chunk | 2.1 M | 6.4 s | 1,576 MB |
| 256×1024×1280 whole volume | 336 M (160×) | 658 s (103×) | 1,576 MB |

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
torch 2.0.0 build is not ABI-compatible with).


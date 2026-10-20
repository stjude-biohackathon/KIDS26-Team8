# Performance

Measured on the St. Jude HPC, 8 worker processes per node, 480 tiles (128³) per volume (3 ROIs × 160).

| Volume | Settings | Wall time | Median per tile (one process) |
|---|---|---|---|
| hippocampus | `--preset hippocampus` | 312 s | 5.2 s |
| PFC | defaults | 863 s (earlier code) | 13.9 s |

Effective per-tile cost is roughly single-process time ÷ 8 workers (≈0.3–0.5 s/tile for the fast runs).

**Comparison with Cellpose-SAM** (figures supplied by the user, GPU): 6.4 s for one 128³ tile, 4.1 s/tile
amortised over 160 tiles, 1.6 GB peak GPU memory. Watershed on 8 CPU workers is roughly an order of magnitude
faster per tile, but Cellpose-SAM needs no per-region tuning; the two are not accuracy-equivalent comparisons.

**Caveats.** A later PFC run reported 135 s, but it was on a different node and the deployed code was not confirmed to be the new version, so it is not a measured speed-up. Cluster nodes differ by up to ~1.5× in speed — compare code versions on one node
(`bsub -m <node>`) and confirm the deployed code is the new one (`grep -c _sparse_peaks segmentation.py` ≥ 1).
The remaining cost is dominated by the distance transform and Gaussian filters; a GPU port is not expected to pay
off yet. Every run writes `timing_per_chunk.csv` and prints a `TIMING:` line.

**Exact speed-ups already in** (identical labels): sparse seed search, lazy solidity. **Opt-in**: `--fast-edt`
(`FAST_EDT=1` in the LSF script).

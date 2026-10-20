# Changelog

## 0.3.0
- QC report (`nuclei_stage1.qc`, `nuclei-qc`): short report with representative and edge images, detailed report of every flagged tile, random audit sheet; run automatically by `scripts/submit_slab.lsf` (`QC=0` to skip).
- Faster, output-identical seed search and solidity screen; opt-in `fast_edt` / `--fast-edt` / `FAST_EDT=1`.
- Per-tile timing (`timing_per_chunk.csv`, `TIMING:` line).
- Docs: QC.md, PERFORMANCE.md; "slab" → "volume" in prose.

## 0.2.0
- Presets, experimental automatic per-tile settings, stitching, viewers.

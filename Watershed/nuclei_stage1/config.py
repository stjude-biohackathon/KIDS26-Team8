"""
Parameters for stage 1. One dataclass; every value below is the default that was tuned on the corrected
prefrontal-cortex ground-truth chunk (see docs/VALIDATION.md for the evidence and its limits).

Override anything at construction time, e.g.  Stage1Config(target_foreground_percentile=92.0), or with
dataclasses.replace(cfg, min_seed_distance=12).
"""
from dataclasses import dataclass


@dataclass
class Stage1Config:
    # ---- geometry ----------------------------------------------------------------------------------
    # Nominal nucleus diameter in voxels. Only used to turn "expected volume" into the size screens below
    # ((4/3) pi (d/2)^3 = ~1767 voxels at 15).
    nucleus_diameter_vox: float = 15.0

    # Physical voxel size in micrometres, (axis0, axis1, axis2) in the tile's NATIVE array order. Axis 0 is the
    # coarse (scan) axis. The magnitudes come from a camera-spec note and are NOT verified against acquisition
    # metadata -- check them for your data. They set the distance transform, the seed-exclusion footprint,
    # the halo size and the length cap, so errors here change results.
    voxel_size: tuple = (2.0, 0.5616, 0.528)

    # ---- foreground mask ---------------------------------------------------------------------------
    threshold_mode: str = "percentile"          # "percentile" (validated) | "global" | "local" (Otsu; comparison only)
    target_foreground_percentile: float = 90.0  # per z-slice, keep the brightest (100 - p) % as candidate foreground
    background_floor_sigma: float = 1.5         # also require voxel > slice median + k * 1.4826 * MAD. 0 = off.
                                                # Stops slices with no nuclei from yielding noise "foreground".
                                                # k <= 1.5 left the corrected chunk unchanged; k = 2 cost F1.
    mask_smoothing_radius: int = 0              # morphological opening(+closing) of the mask. 0 = off. It hurt every
    mask_smoothing_closing: bool = True         # metric when tested (radius 1-2); leave off.

    # EXPERIMENTAL, off by default: build the mask from a difference-of-Gaussians (band-pass) image instead of the raw
    # per-slice percentile. (lo, hi) are Gaussian sigmas in um: G(lo) - G(hi), then ONE global percentile cutoff
    # (target_foreground_percentile). Helps where nucleus and background brightness overlap (hippocampus chunk:
    # F1 0.56 -> 0.80). See PRESETS below and docs/VALIDATION.md.
    mask_bandpass_um: tuple = None

    # ---- seeds + watershed -------------------------------------------------------------------------
    fast_edt: bool = False   # opt-in speed-up: use the `edt` package (pip install edt) for the distance transform, ~3x faster, float32 (~1e-7 differences; labels identical on all our ground-truth chunks but not guaranteed). Off = scipy, exact.
    min_seed_distance: int = 17                 # seed exclusion radius, in voxels of the FINEST axis (see
                                                # segmentation._anisotropic_seed_footprint); ~ a nucleus radius
    seed_prominence: float = 0.15               # peak_local_max threshold_rel. Inert in the range 0.10-0.15.
    seed_landscape_smooth_um: float = 0.0       # Gaussian sigma (um) on the distance map used for seeding/flooding.
                                                # 0 = off. 1.0-1.5 with min_seed_distance 10-12 scored better on the
                                                # one corrected chunk (F1 0.82-0.84 vs 0.796) but is not default
                                                # until confirmed on a second chunk.
    min_object_voxels: int = 100                # drop specks below this after watershed

    # ---- instance screen (what ends up in `accepted`) ----------------------------------------------
    solidity_thresh: float = 0.15               # flag if solidity below this (a very loose bound: most watershed
                                                # objects here have low solidity)
    volume_zscore_thresh: float = 4.0           # flag if |volume z-score| vs the other objects in the block exceeds
                                                # this
    min_volume_fraction_of_expected: float = 0.20   # flag if volume < this fraction of the nominal nucleus volume
    max_volume_fraction_of_expected: float = 0.0    # flag if volume > this multiple of nominal. 0 = off (best known)
    accept_big_side_flags: bool = True          # keep instances flagged ONLY as too big / positive volume outlier
    max_extent_um: float = 60.0                 # flag as "too_long" if the bounding box exceeds this in any axis
                                                # (physical units). 0 = off. Catches columns / streaks.


# Named parameter sets (nuclei_stage1.run_slab --preset NAME; scripts/submit_slab.lsf: PRESET=NAME). The default
# Stage1Config is the PFC-validated one. A preset is for ONE tissue type: apply it only to slabs of that tissue.
PRESETS = {
    # Tuned against ONE corrected hippocampus chunk (74 nuclei) alone; flat top (a dozen neighbouring settings are
    # within ~0.03 F1), so treat the exact numbers as soft. F1 0.558 -> 0.800 on that chunk (tuned and scored on the
    # same nuclei, so optimistic).
    "hippocampus": dict(mask_bandpass_um=(2.0, 8.0), target_foreground_percentile=94.0, min_seed_distance=12,
                        seed_landscape_smooth_um=1.0),
}

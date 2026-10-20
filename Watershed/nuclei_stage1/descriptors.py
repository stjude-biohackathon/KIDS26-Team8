"""
Cheap image statistics of a raw chunk, used to pick parameters from what the chunk LOOKS like (see autoconfig.py), not
from a brain-region label (labels can be wrong or hide heterogeneity inside a slab).

IMPORTANT: "fraction of bright foreground" must NOT be measured with the pipeline's own percentile threshold.
target_foreground_percentile=90 marks the brightest 10 % of every slice as foreground BY CONSTRUCTION, so that fraction
is a constant and tells you nothing about density. These descriptors are anchored to the chunk's own background level
(robust median + k * MAD) or to detected blobs, so they move when real nucleus density moves.

Cost: ~0.3 s per 128^3 chunk, ~3 s per 256^3.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

MAD_TO_SIGMA = 1.4826


def describe_chunk(vol, nucleus_diameter_vox=15.0):
    vol = np.asarray(vol, dtype=np.float32)
    nonzero = vol[vol > 0]                      # ignore exact-zero edge padding
    zero_frac = float((vol == 0).mean())
    if nonzero.size < 1000:
        return {"zero_frac": zero_frac, "usable": False}

    med = float(np.median(nonzero))
    mad = float(np.median(np.abs(nonzero - med))) * MAD_TO_SIGMA or 1e-6

    d = {"zero_frac": zero_frac, "usable": True, "median": med, "robust_sigma": mad}

    # Foreground fraction relative to the chunk's own background level.
    # Several k values: nuclei in weak-contrast data may only clear a low k.
    valid = vol > 0
    for k in (2, 3, 5):
        d[f"fg_frac_k{k}"] = float(((vol > med + k * mad) & valid).sum() / valid.sum())

    # Contrast: how far the bright tail sits above background, in sigma units.
    d["contrast_p99"] = float((np.percentile(nonzero, 99) - med) / mad)

    # Blob density: peaks of the smoothed volume that stand clearly above
    # background, per megavoxel. Independent of any watershed parameter.
    sigma = nucleus_diameter_vox / 5.0
    sm = ndi.gaussian_filter(vol, sigma)
    sm_med = float(np.median(sm[valid]))
    sm_mad = float(np.median(np.abs(sm[valid] - sm_med))) * MAD_TO_SIGMA or 1e-6
    peaks = peak_local_max(sm, min_distance=int(nucleus_diameter_vox // 3),
                            threshold_abs=sm_med + 3 * sm_mad, exclude_border=False)
    d["blobs_per_mvox"] = float(len(peaks) / (valid.sum() / 1e6))
    d["n_blobs"] = int(len(peaks))
    return d

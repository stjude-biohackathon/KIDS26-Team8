"""
Stage 1 segmentation: per-slice percentile foreground mask -> distance transform -> seeded watershed ->
per-instance quality screen.

    run_stage1(raw_block, cfg, valid=None) -> (accepted, difficult, labels)

  accepted   label volume of instances that passed the screen (this is the pipeline's output)
  difficult  boolean mask of instances that were flagged (solidity / size / length), kept for inspection
  labels     the raw watershed labels before screening

Everything here is CPU-only and works on one array at a time (a 128^3 chunk, or a chunk plus halo).
See docs/METHODS.md for what each step does and docs/VALIDATION.md for the evidence behind the defaults.

Otsu thresholding ("local"/"global" modes) is kept only for comparison: on this data it marked 29-46 % of a
chunk as foreground when real nuclei occupy ~3-5 %, because the intensity contrast is too weak for a bimodal
histogram. The default "percentile" mode is the one that was tuned and validated.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_local
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import ball, opening, closing


def _bg_floor(values, k):
    """median + k * robust sigma of the non-zero values (exact-zero = padding, ignored)."""
    nz = values[values > 0]
    if nz.size < 100:
        return -np.inf
    med = np.median(nz)
    return float(med + k * 1.4826 * np.median(np.abs(nz - med)))


def foreground_mask(raw, mode="percentile", block_size=51, target_foreground_percentile=96.0,
                     use_tophat=False, tophat_radius=8, valid=None, background_floor_sigma=0.0):
    """Binarize a raw intensity block. Formerly named local_otsu_mask --
    renamed since "percentile" (the default) has nothing to do with Otsu;
    see this module's docstring for why.

    use_tophat: optionally apply a white top-hat filter (suppresses
    slowly-varying background, enhances small bright blobs ~tophat_radius
    in size) before thresholding. Direct testing on real data found this
    did NOT improve foreground fraction at all versus plain global Otsu
    (29.12% vs 28.95%, effectively identical) while costing ~11s/chunk
    (a ball(8) structuring element on a 128^3 volume) -- kept available
    for comparison, but not expected to help, and meaningfully slower.

    "percentile" (default, recommended) marks the brightest
    (100 - target_foreground_percentile)% of voxels in EACH z-slice as
    foreground. This directly and robustly targets real nucleus density
    rather than relying on Otsu's assumption of a clean bimodal
    histogram -- confirmed via direct testing on real data that Otsu
    ("local" or "global" below) was marking 29-46% of a chunk
    foreground when the ilastik pixel classifier we compared against considered only
    ~3-5% real signal, because this data's raw-intensity contrast is
    too weak for a genuine histogram break to exist. Per-slice (not one
    cutoff for the whole 3D block) preserves adaptivity to
    depth-dependent illumination falloff, the same property "local"
    Otsu mode was trying to achieve.

    "local" (legacy, actual Otsu) uses a per-slice ADAPTIVE threshold
    (each pixel compared to its own local neighborhood average) -- on
    noisy, weakly-contrasted data this behaves more like a texture/edge
    detector than a true foreground separator (roughly half of any
    textured neighborhood ends up "above its own local average" by
    construction), which is what produced the 46% foreground fraction.
    "global" (legacy, actual Otsu) is a single Otsu threshold for the
    whole block; better than "local" here (29%) but still far above
    real density. Neither legacy Otsu mode is recommended for this
    dataset -- kept only so you can re-compare against percentile
    thresholding on your own data if you ever want to double-check.

    valid: optional boolean array, same shape as raw, True where a voxel is REAL data.
    Needed for deskewed chunks (deskew.py), whose shear leaves a wedge of zero padding in
    every z-slice: a percentile taken over the whole plane would count those padding zeros,
    quietly lowering the cutoff by up to ~1.3 points in the most-shifted slices. With
    `valid`, the percentile uses real voxels only and padding is never foreground. Left as
    None (every voxel real) the behaviour is exactly as before. Fully honoured in
    "percentile" mode; the other modes only have padding removed from their output.
    """
    if mode == "percentile":
        # background_floor_sigma > 0: in addition to the per-slice percentile, a voxel must exceed that
        # slice's robust background (median + k * 1.4826 * MAD of its real, non-zero voxels). The
        # percentile alone ALWAYS marks the brightest N% of a slice as foreground, so a slice with no
        # nuclei still yields "foreground" made of noise / stripes. 0 = off (exact prior behaviour).
        if use_tophat:
            from skimage.morphology import white_tophat, ball
            raw = white_tophat(raw, ball(tophat_radius))
        mask = np.zeros_like(raw, dtype=bool)
        for z in range(raw.shape[0]):
            plane = raw[z]
            if valid is None:
                if plane.max() == plane.min():
                    continue
                t = np.percentile(plane, target_foreground_percentile)
                if background_floor_sigma > 0:
                    t = max(t, _bg_floor(plane, background_floor_sigma))
                mask[z] = plane > t
            else:
                v = valid[z]
                if not v.any():
                    continue
                real = plane[v]
                if real.max() == real.min():
                    continue
                t = np.percentile(real, target_foreground_percentile)
                if background_floor_sigma > 0:
                    t = max(t, _bg_floor(real, background_floor_sigma))
                mask[z] = (plane > t) & v
        return mask

    if mode == "global":
        if use_tophat:
            from skimage.morphology import white_tophat, ball
            raw = white_tophat(raw, ball(tophat_radius))
        t = threshold_otsu(raw if valid is None else raw[valid])
        out = raw > t
        return out if valid is None else out & valid

    mask = np.zeros_like(raw, dtype=bool)
    for z in range(raw.shape[0]):
        plane = raw[z]
        if plane.max() == plane.min():
            continue
        local_t = threshold_local(plane, block_size=block_size, method="gaussian")
        mask[z] = plane > local_t
    return mask if valid is None else mask & valid


def _anisotropic_seed_footprint(min_seed_distance, voxel_size):
    """A binary structuring element for peak_local_max's `footprint`
    argument that is a true sphere of radius min_seed_distance in PHYSICAL
    space, not voxel-index space -- i.e. an ellipsoid in voxel counts when
    voxel_size is anisotropic.

    Why this exists: peak_local_max's plain `min_distance=` scalar
    excludes seeds using a radius in ARRAY-INDEX units, applied equally
    on every axis, regardless of voxel_size. That was an exact match for
    an isotropic (1,1,1) placeholder (the calibration min_seed_distance=10
    was tuned under), but once voxel_size is real and anisotropic, a
    10-voxel-index exclusion radius means wildly different PHYSICAL
    distances per axis -- confirmed to cause real under-segmentation
    (more merged_pred_objects, lower recall) on real ground truth once
    real voxel dimensions (X ~3.6-3.8x coarser than Y/Z) were wired in.

    Radius in voxels along axis i = min_seed_distance * min(voxel_size) /
    voxel_size[i] -- i.e. min_seed_distance voxels on whichever axis is
    FINEST (this preserves the already-tuned value exactly on that axis),
    scaled down on coarser axes so the exclusion region is the same
    physical size in every direction. When voxel_size is exactly (1,1,1)
    this reduces to a plain isotropic ball of radius min_seed_distance --
    byte-identical to the old behavior, confirmed by test."""
    finest = min(voxel_size)
    radii_vox = [max(1, int(round(min_seed_distance * finest / v))) for v in voxel_size]
    zz, yy, xx = np.meshgrid(
        np.arange(-radii_vox[0], radii_vox[0] + 1),
        np.arange(-radii_vox[1], radii_vox[1] + 1),
        np.arange(-radii_vox[2], radii_vox[2] + 1),
        indexing="ij",
    )
    physical_r2 = (zz * voxel_size[0]) ** 2 + (yy * voxel_size[1]) ** 2 + (xx * voxel_size[2]) ** 2
    R = min_seed_distance * finest
    return physical_r2 <= R * R


def _ellipsoid_max_filter(a, footprint):
    """Exact equivalent of scipy.ndimage.maximum_filter(a, footprint=footprint, mode='nearest') for a
    footprint made of contiguous, centred rows along the last axis (true for the ellipsoid built by
    _anisotropic_seed_footprint). A maximum over a union of rows is the maximum of the row maxima, so each
    distinct row width is computed once with a 1-D running maximum and the (dz, dy) shifts are combined with
    np.maximum. Maxima involve no arithmetic, so the result is bit-identical to scipy's; it is about an
    order of magnitude faster than the general N-D filter for large footprints. Returns None if the
    footprint is not of that form (caller then uses scipy)."""
    fp = np.asarray(footprint, dtype=bool)
    if fp.ndim != 3 or any(n % 2 == 0 for n in fp.shape):
        return None
    rz, ry, rx = (n // 2 for n in fp.shape)
    widths = {}
    for iz in range(fp.shape[0]):
        for iy in range(fp.shape[1]):
            xs = np.nonzero(fp[iz, iy])[0]
            if len(xs) == 0:
                continue
            lo, hi = xs[0] - rx, xs[-1] - rx
            if hi - lo + 1 != len(xs) or lo != -hi:
                return None
            widths[(iz - rz, iy - ry)] = int(hi)
    nz, ny, nx = a.shape
    pad = np.pad(a, ((rz, rz), (ry, ry), (rx, rx)), mode="edge")   # == mode='nearest'
    line_max = {}
    out = None
    for (dz, dy), w in widths.items():
        if w not in line_max:
            line_max[w] = ndi.maximum_filter1d(pad, size=2 * w + 1, axis=2, mode="nearest") if w else pad
        sl = line_max[w][rz + dz: rz + dz + nz, ry + dy: ry + dy + ny, rx: rx + nx]
        if out is None:
            out = sl.copy()
        else:
            np.maximum(out, sl, out=out)
    return out


def _sparse_peaks(img, A, footprint):
    """Exact peak test restricted to the candidate voxels A (img above threshold, inside the mask).
    A candidate p is a peak iff no voxel in its footprint is strictly larger. Voxels not in A are below
    the threshold, hence below every candidate, so they can be ignored. Two cheap separable box maxima
    bracket the true footprint maximum E:  L (largest box inside the footprint) <= E <= U (box around it).
    img == U  -> peak;  img < L -> not a peak;  only the few voxels in between (typically < 1% of A) are
    checked exactly, offset by offset. Maxima involve no arithmetic, so the answer is exact.
    Returns a boolean volume of peaks, or None if the footprint is not a centred, symmetric blob."""
    fp = np.asarray(footprint, dtype=bool)
    if fp.ndim != 3 or any(n % 2 == 0 for n in fp.shape) or not fp[tuple(n // 2 for n in fp.shape)]:
        return None
    if not np.array_equal(fp, fp[::-1, ::-1, ::-1]):
        return None
    r = tuple(n // 2 for n in fp.shape)
    mn = np.finfo(img.dtype).min
    img2 = np.where(A, img, mn)

    def box_max(h):
        out = img2
        for ax, hh in enumerate(h):
            if hh:
                out = ndi.maximum_filter1d(out, 2 * hh + 1, axis=ax, mode="constant", cval=mn)
        return out

    U = box_max(r)
    inner = (0, 0, 0)
    for sc in np.linspace(1.0, 0.0, 101):                 # largest scaled box lying entirely inside the footprint
        h = tuple(int(np.floor(sc * rr)) for rr in r)
        if fp[r[0] - h[0]: r[0] + h[0] + 1, r[1] - h[1]: r[1] + h[1] + 1, r[2] - h[2]: r[2] + h[2] + 1].all():
            inner = h
            break
    L = box_max(inner) if any(inner) else img2

    peaks = A & (img2 == U)
    amb = A & ~peaks & (img2 >= L)
    if int(amb.sum()) > 5000:
        # many undecided voxels (flat plateaus / near-uniform maps): the exact per-offset check would be
        # slower than one dense row-decomposed filter, so use that (still exact)
        E = _ellipsoid_max_filter(img2, fp)
        return None if E is None else (A & (img2 == E))
    if amb.any():
        c = np.argwhere(amb)
        v = img2[amb]
        pad = np.pad(img2, [(rr, rr) for rr in r], mode="constant", constant_values=mn)
        offs = np.argwhere(fp) - np.array(r)
        inside = np.all(np.abs(offs) <= np.array(inner), axis=1)     # these cannot exceed v (v >= L)
        exceeded = np.zeros(len(c), dtype=bool)
        for o in offs[~inside]:
            q = pad[c[:, 0] + r[0] + o[0], c[:, 1] + r[1] + o[1], c[:, 2] + r[2] + o[2]]
            exceeded |= q > v
        keep = c[~exceeded]
        peaks[tuple(keep.T)] = True
    return peaks


def _fast_peak_coords(distance, footprint, mask, seed_prominence):
    """Same peaks as skimage's peak_local_max(distance, footprint=footprint, labels=mask,
    exclude_border=False, threshold_rel=seed_prominence) (single label, no spacing step), found with
    _sparse_peaks instead of scipy's dense N-D maximum_filter (about 10x faster here). Falls back to
    skimage whenever the shortcut does not apply, so the output never differs."""
    def fallback():
        return peak_local_max(distance, footprint=footprint, labels=mask,
                              exclude_border=False, threshold_rel=seed_prominence)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or distance.dtype.kind != "f":
        return fallback()
    roi = ndi.find_objects(mask.astype(int))[0]
    lab = mask[roi]
    img = distance[roi].copy()
    img[~lab] = np.finfo(img.dtype).min
    threshold = max(distance.min(), seed_prominence * distance.max()) if seed_prominence is not None \
        else distance.min()
    A = (img > threshold) & lab
    if not A.any():
        return fallback()
    out = _sparse_peaks(img, A, footprint)
    # skimage special-cases a "trivial" image (every masked voxel is a peak); if every candidate is a peak
    # we cannot rule that out cheaply, so defer to skimage in that rare case
    if out is None or int(out.sum()) == int(A.sum()):
        return fallback()
    coords = np.argwhere(out)
    for i, sl in enumerate(roi):
        coords[:, i] += sl.start
    return coords


def _edt(mask, voxel_size, fast=False):
    """Euclidean distance transform inside `mask` (distance to the nearest background voxel, physical units).
    fast=True uses the `edt` package (pip install edt): ~3x faster than scipy single-threaded, but it computes in
    float32, so values differ from scipy's by ~1e-7. That was invisible in our tests (labels identical on all
    ground-truth chunks) but is not guaranteed bit-identical, hence opt-in. Falls back to scipy if `edt` is missing."""
    if fast:
        try:
            import edt
            return edt.edt(np.asarray(mask, dtype=bool), anisotropy=tuple(float(v) for v in voxel_size),
                           black_border=False, parallel=1).astype(np.float64)
        except ImportError:
            print("fast_edt requested but the 'edt' package is not installed (pip install edt); using scipy", flush=True)
    return ndi.distance_transform_edt(mask, sampling=voxel_size)


def seeded_watershed(raw, mask, min_seed_distance, min_object_voxels, seed_prominence=0.3,
                      voxel_size=(1.0, 1.0, 1.0), mask_smoothing_radius=0,
                      mask_smoothing_closing=True, seed_landscape_smooth_um=0.0, fast_edt=False):
    # Morphological opening(+closing) on the mask BEFORE the distance
    # transform -- jagged/noisy mask edges create spurious local peaks
    # in the distance transform, which peak_local_max then misreads as
    # separate nuclei. This targets that specific mechanism directly
    # (cleaning up mask geometry) rather than blurring the underlying
    # raw signal everything else depends on. radius=0 is an exact no-op
    # matching prior behavior.
    #
    # mask_smoothing_closing: real-ground-truth testing found opening+
    # closing together (radius 1 or 2) hurt EVERY metric vs no smoothing
    # at all -- recall collapsed hardest, suggesting CLOSING (which
    # bridges nearby regions) was merging/eroding genuinely separate,
    # closely-packed real nuclei rather than just cleaning up noise.
    # Opening alone never bridges separate objects -- only closing does
    # -- so this flag lets opening-only be tested as a gentler
    # alternative before concluding mask smoothing doesn't help at all
    # on this data.
    if mask_smoothing_radius > 0:
        footprint = ball(mask_smoothing_radius)
        mask = opening(mask, footprint)
        if mask_smoothing_closing:
            mask = closing(mask, footprint)

    # sampling=voxel_size makes the distance transform physically correct
    # if voxels are anisotropic (e.g. coarser z-spacing) -- watershed floods
    # based on this map's actual VALUES, so an isotropic assumption on
    # anisotropic data would place split boundaries wrong even with correct
    # seed placement. Defaults to (1,1,1), an exact no-op matching prior
    # behavior until real voxel dimensions are confirmed.
    distance = _edt(mask, voxel_size, fast_edt)
    # Optional: Gaussian-smooth the distance map (sigma in physical units, converted to voxels per
    # axis) before seeds are found and the watershed floods. The MASK is untouched, so object
    # boundaries still come from the unsmoothed raw. This removes spurious distance-map peaks
    # (-> fewer over-split nuclei) and lets a smaller min_seed_distance work. Measured on the
    # corrected PFC chunk (sweep_seed_landscape_against_gt.py): sigma 1.0-1.5 um with
    # min_seed_distance 10-12 gave F1 0.82-0.84 and IoU 0.62-0.63 vs 0.796/0.602 (on the current labels) for the
    # unsmoothed landscape at 17. 0 = exact no-op (prior behaviour).
    if seed_landscape_smooth_um and seed_landscape_smooth_um > 0:
        distance = ndi.gaussian_filter(
            distance, tuple(seed_landscape_smooth_um / v for v in voxel_size))
    # Anisotropic voxels need the seed-exclusion region to be a sphere in
    # PHYSICAL space (an ellipsoid in voxel counts), not a plain isotropic
    # radius in array-index space -- see _anisotropic_seed_footprint's
    # docstring. voxel_size==(1,1,1) takes the old scalar min_distance
    # path exactly, so nothing changes for any isotropic-voxel_size run.
    if tuple(voxel_size) == (1.0, 1.0, 1.0):
        coords = peak_local_max(
            distance, min_distance=min_seed_distance, labels=mask,
            exclude_border=False, threshold_rel=seed_prominence,
        )
    else:
        footprint = _anisotropic_seed_footprint(min_seed_distance, voxel_size)
        coords = _fast_peak_coords(distance, footprint, mask, seed_prominence)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        seed_mask[tuple(coords.T)] = True
    markers = cc_label(seed_mask)
    labels = watershed(-distance, markers, mask=mask)

    # Drop instances below min_object_voxels WITHOUT re-merging the
    # survivors by connectivity: watershed-separated instances that
    # touch each other are still connected in the binary mask, so
    # cc_label(labels > 0) here would silently re-merge every touching
    # instance back into one blob whenever even a single small speck
    # needed dropping -- undoing the entire point of watershed.
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_object_voxels
    if len(keep):
        keep[0] = False
    ids_to_drop = np.nonzero(~keep)[0]
    if len(ids_to_drop):
        labels[np.isin(labels, ids_to_drop)] = 0

    remaining = np.unique(labels)
    remaining = remaining[remaining != 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    remap[remaining] = np.arange(1, len(remaining) + 1, dtype=labels.dtype)
    labels = remap[labels]
    return labels


def _solidity_bound_clears(image, area, thr):
    """True only if the object's solidity is PROVEN to be >= thr without computing its 3-D convex hull.
    The hull's projection along any axis equals the 2-D hull of the object's projection, so the 3-D hull lies
    inside the intersection of the three extruded 2-D hulls; counting that intersection gives an upper bound V
    on the hull volume, hence solidity >= area / V. A safety margin (2% + 10 voxels) keeps it conservative, so
    near-threshold objects always get the exact hull. Any failure (flat or degenerate shape) returns False."""
    try:
        from skimage.morphology import convex_hull_image
        image = np.asarray(image, dtype=bool)
        hz = convex_hull_image(image.any(axis=0))      # (y, x)
        hy = convex_hull_image(image.any(axis=1))      # (z, x)
        hx = convex_hull_image(image.any(axis=2))      # (z, y)
        v = int(np.count_nonzero(hz[None, :, :] & hy[:, None, :] & hx[:, :, None]))
        return area / (v * 1.02 + 10) >= thr
    except Exception:
        return False


def explain_instances(labels, expected_diameter, solidity_thresh, volume_zscore_thresh,
                       min_volume_fraction_of_expected=0.35, max_volume_fraction_of_expected=0.0,
                       voxel_size=(1.0, 1.0, 1.0), max_extent_um=0.0, lazy_solidity=False):
    """Per-instance report behind classify_instances: for every instance, its volume,
    solidity, volume z-score, and the list of reasons it would be flagged difficult
    (any of "solidity", "volume_zscore", "too_small", "too_big", "too_long"; empty list = accepted).
    "too_long": the instance's longest bounding-box extent, in physical units (voxel_size), exceeds
    max_extent_um (0 = off). Catches columns / streaks that span many planes -- long, thin objects are
    convex and not especially big, so no other rule sees them. Never promoted by accept_big_side_flags.
    classify_instances is built on this, so the reasons can never drift from the
    actual decision."""
    expected_vol = (4 / 3) * np.pi * (expected_diameter / 2) ** 3
    props = regionprops(labels)
    vols = np.array([p.area for p in props]) if props else np.array([])
    if len(vols) > 2:
        mu, sigma = vols.mean(), vols.std() + 1e-6
    else:
        mu, sigma = expected_vol, expected_vol * 0.5

    out = []
    for p in props:
        z = (p.area - mu) / sigma
        reasons = []
        if lazy_solidity:
            # solidity = area / convex-hull area and the hull lies inside the bounding box, so
            # solidity >= area / bbox volume. If that bound already clears the threshold the (slow)
            # convex hull cannot change the decision, so skip it; the reported value is then the bound.
            b = p.bbox
            bbox_vol = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
            sol = p.area / bbox_vol
            if sol < solidity_thresh:
                if _solidity_bound_clears(p.image, p.area, solidity_thresh):
                    sol = solidity_thresh
                else:
                    sol = p.solidity
        else:
            sol = p.solidity
        if sol < solidity_thresh:
            reasons.append("solidity")
        if abs(z) > volume_zscore_thresh:
            reasons.append("volume_zscore")
        if p.area < expected_vol * min_volume_fraction_of_expected:
            reasons.append("too_small")
        if max_volume_fraction_of_expected > 0 and p.area > expected_vol * max_volume_fraction_of_expected:
            reasons.append("too_big")
        if max_extent_um > 0:
            b = p.bbox
            longest = max((b[k + 3] - b[k]) * voxel_size[k] for k in range(3))
            if longest > max_extent_um:
                reasons.append("too_long")
        out.append({"label": p.label, "volume": int(p.area), "solidity": float(sol),
                    "zscore": float(z), "reasons": reasons})
    return out


def _big_side_only(reasons, zscore):
    """True if an instance's flag reasons are EXCLUSIVELY too_big and/or a
    volume_zscore outlier on the positive (implausibly large) side. Used
    by accept_big_side_flags to decide which flagged instances get kept
    anyway -- see classify_instances and is_effectively_accepted."""
    reasons = set(reasons)
    if not reasons:
        return False
    return reasons <= {"too_big", "volume_zscore"} and ("volume_zscore" not in reasons or zscore > 0)


def is_effectively_accepted(entry, accept_big_side_flags):
    """Given one explain_instances() report entry, decide whether it
    counts as accepted under the given accept_big_side_flags policy.
    classify_instances is built on this (same pattern as
    explain_instances/classify_instances above), and any other script
    reporting on what got kept vs. discarded (run_stage1_filtered.py's
    FLAGGED DIFFICULT report, --save-difficult) should use this too, so
    the two can never disagree about which instances were actually kept."""
    if not entry["reasons"]:
        return True
    return accept_big_side_flags and _big_side_only(entry["reasons"], entry["zscore"])


def classify_instances(labels, expected_diameter, solidity_thresh, volume_zscore_thresh,
                        min_volume_fraction_of_expected=0.35,
                        max_volume_fraction_of_expected=0.0,
                        accept_big_side_flags=False,
                        voxel_size=(1.0, 1.0, 1.0), max_extent_um=0.0):
    """Return (accepted_labels, difficult_mask).

    accepted_labels: label volume containing only confidently-segmented
                      instances, with final label ids preserved.
    difficult_mask:   binary mask of everything else, to be re-cropped
                       and set aside for inspection.

    accept_big_side_flags: see Stage1Config.accept_big_side_flags --
    when True, an instance flagged ONLY too_big and/or a positive
    volume_zscore is still returned as accepted rather than difficult, no
    geometry touched. too_small and solidity flags are never promoted.

    THREE independent difficulty signals are used:
      - solidity: real nuclei are close to convex; a low-solidity blob
        after watershed usually means an unresolved merge or a bad split
        (watershed "succeeded" but produced the wrong shape).
      - volume z-score against this block's own instance population:
        catches objects that are implausibly large (probably still-merged
        clumps) or small (probably fragments), relative to LOCAL context.
      - min_volume_fraction_of_expected: an ABSOLUTE size floor, separate
        from the z-score. The z-score only flags instances that are
        outliers relative to their own chunk's population -- a chunk
        uniformly over-segmented into many similarly-small, individually
        solid fragments passes right through it, since every fragment
        looks "normal" next to its equally-small neighbors even though
        all of them are absolutely too small to be a whole real nucleus.
        Confirmed directly: without this check, 64 simulated uniform
        small fragments were 100% "accepted" (0 routed to stage 2).
      - max_volume_fraction_of_expected: the mirror-image ABSOLUTE
        ceiling (0 = off). The z-score is measured against this chunk's
        own population, which is inflated by the very oversized blobs
        it is meant to catch: on a real chunk, instances of 5000-7000
        voxels (3-4x the expected ~1767) scored only z=3-4.2 because
        they had blown up the std. Solidity can't catch them either --
        nearly every watershed instance in this data has low solidity.
        An absolute cap on volume doesn't depend on either.
    """
    # lookup tables instead of one full-volume `labels == id` per instance (that was O(n_instances x voxels)
    # and dominated runtime on large/padded blocks)
    top = int(labels.max()) + 1
    acc_lut = np.zeros(top, dtype=labels.dtype)
    diff_lut = np.zeros(top, dtype=bool)
    for e in explain_instances(labels, expected_diameter, solidity_thresh, volume_zscore_thresh,
                                min_volume_fraction_of_expected, max_volume_fraction_of_expected,
                                voxel_size, max_extent_um, lazy_solidity=True):
        if is_effectively_accepted(e, accept_big_side_flags):
            acc_lut[e["label"]] = e["label"]
        else:
            diff_lut[e["label"]] = True

    return acc_lut[labels], diff_lut[labels]


def bandpass_mask(raw, voxel_size, lo_um, hi_um, percentile, valid=None):
    """Foreground mask from a difference-of-Gaussians image, G(lo_um) - G(hi_um), thresholded by ONE global percentile.
    Removes slowly varying background so nuclei stand out when nucleus and background intensities overlap (on the
    hippocampus chunk the mask's Dice against corrected labels went 0.43 raw -> 0.65). The band-pass image has no
    per-slice illumination trend left, hence a global (whole-block) percentile rather than a per-slice one; the
    consequence is that an empty block still gets ~(100 - percentile) % foreground. Used only to decide WHICH voxels
    are foreground: seeds and boundaries still come from this mask, not from the filtered image. Experimental:
    opt in with cfg.mask_bandpass_um=(lo, hi)."""
    v = np.asarray(voxel_size, float)
    r = raw.astype(np.float32)
    img = ndi.gaussian_filter(r, lo_um / v) - ndi.gaussian_filter(r, hi_um / v)
    if valid is None:
        return img > np.percentile(img, percentile)
    if not valid.any():
        return np.zeros(raw.shape, bool)
    return (img > np.percentile(img[valid], percentile)) & valid


def run_stage1(raw_block, cfg, valid=None):
    """valid: optional boolean mask of real (non-padding) voxels -- see foreground_mask.
    Pass it for deskewed chunks; omit it otherwise (behaviour unchanged)."""
    bp = getattr(cfg, "mask_bandpass_um", None)
    if bp:
        mask = bandpass_mask(raw_block, cfg.voxel_size, bp[0], bp[1], cfg.target_foreground_percentile, valid=valid)
    else:
        mask = foreground_mask(
            raw_block, mode=cfg.threshold_mode,
            target_foreground_percentile=cfg.target_foreground_percentile,
            valid=valid,
            background_floor_sigma=getattr(cfg, "background_floor_sigma", 0.0),
        )
    labels = seeded_watershed(
        raw=raw_block, mask=mask,
        min_seed_distance=cfg.min_seed_distance,
        min_object_voxels=cfg.min_object_voxels,
        seed_prominence=cfg.seed_prominence,
        voxel_size=cfg.voxel_size,
        mask_smoothing_radius=cfg.mask_smoothing_radius,
        mask_smoothing_closing=cfg.mask_smoothing_closing,
        seed_landscape_smooth_um=getattr(cfg, "seed_landscape_smooth_um", 0.0),
        fast_edt=getattr(cfg, "fast_edt", False),
    )
    accepted, difficult = classify_instances(
        labels, cfg.nucleus_diameter_vox,
        cfg.solidity_thresh, cfg.volume_zscore_thresh,
        cfg.min_volume_fraction_of_expected,
        cfg.max_volume_fraction_of_expected,
        accept_big_side_flags=getattr(cfg, "accept_big_side_flags", False),
        voxel_size=cfg.voxel_size,
        max_extent_um=getattr(cfg, "max_extent_um", 0.0),
    )
    # returns the raw watershed labels too (not just accepted/difficult)
    # so callers that want a fallback for flagged regions (e.g. to run a different segmenter on them later)
    # still have the original shapes available.
    return accepted, difficult, labels

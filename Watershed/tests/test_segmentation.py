from dataclasses import replace

import numpy as np

from helpers import add_ellipsoid, noisy
from nuclei_stage1 import Stage1Config, run_stage1
from nuclei_stage1.segmentation import classify_instances

ISO = replace(Stage1Config(), voxel_size=(1.0, 1.0, 1.0), min_seed_distance=6, nucleus_diameter_vox=12.0,
              target_foreground_percentile=90.0)


def n_objects(lab):
    return len(np.unique(lab)) - (1 if (lab == 0).any() else 0)


def test_separate_spheres_are_counted_once_each():
    v = noisy((48, 64, 64))
    for c in [(24, 16, 16), (24, 16, 48), (24, 48, 32)]:
        add_ellipsoid(v, c, (6, 6, 6), 1000.0)
    accepted, difficult, labels = run_stage1(v, ISO)
    assert n_objects(accepted) == 3


def test_touching_spheres_are_split():
    v = noisy((48, 64, 64))
    add_ellipsoid(v, (24, 32, 22), (7, 7, 7), 1000.0)
    add_ellipsoid(v, (24, 32, 38), (7, 7, 7), 1000.0)
    accepted, _, _ = run_stage1(v, ISO)
    assert n_objects(accepted) == 2


def test_all_zero_block_gives_no_objects():
    accepted, difficult, labels = run_stage1(np.zeros((16, 32, 32), np.float32), ISO)
    assert accepted.max() == 0 and not difficult.any()


def test_classify_matches_a_per_instance_reference():
    """The lookup-table implementation must equal the straightforward one-mask-per-instance version."""
    from nuclei_stage1.segmentation import explain_instances, is_effectively_accepted
    rng = np.random.default_rng(1)
    lab = np.zeros((40, 60, 60), np.int32)
    zz, yy, xx = np.indices(lab.shape)
    for i in range(1, 25):
        c = [rng.integers(5, s - 5) for s in lab.shape]; r = rng.integers(2, 8)
        lab[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r] = i
    args = (12.0, 0.15, 2.0, 0.2, 0.0)
    kw = dict(accept_big_side_flags=True, voxel_size=(1.0, 1.0, 1.0), max_extent_um=0.0)
    acc, diff = classify_instances(lab, *args, **kw)
    ref_acc = np.zeros_like(lab); ref_diff = np.zeros(lab.shape, bool)
    for e in explain_instances(lab, *args, voxel_size=kw["voxel_size"], max_extent_um=0.0):
        if is_effectively_accepted(e, True):
            ref_acc[lab == e["label"]] = e["label"]
        else:
            ref_diff[lab == e["label"]] = True
    assert np.array_equal(acc, ref_acc) and np.array_equal(diff, ref_diff)


def test_too_long_streak_is_flagged_not_accepted():
    lab = np.zeros((30, 40, 40), np.int32)
    lab[5:25, 10:14, 10:14] = 1                    # 20 voxels long
    lab[5:9, 20:30, 20:30] = 2                     # compact blob
    acc, diff = classify_instances(lab, 6.0, 0.0, 99.0, 0.0, 0.0, accept_big_side_flags=True,
                                   voxel_size=(2.0, 1.0, 1.0), max_extent_um=30.0)   # 20 vox * 2 um = 40 um > 30
    assert (acc == 1).sum() == 0 and diff[lab == 1].all()
    assert (acc == 2).sum() == (lab == 2).sum()


def test_background_floor_suppresses_noise_only_slices():
    noise = noisy((8, 64, 64))
    from nuclei_stage1.segmentation import foreground_mask
    off = foreground_mask(noise, target_foreground_percentile=90.0, background_floor_sigma=0.0)
    on = foreground_mask(noise, target_foreground_percentile=90.0, background_floor_sigma=1.5)
    assert off.mean() > 0.09 and on.mean() < off.mean()


def test_fast_seed_search_matches_skimage():
    """_fast_peak_coords must return exactly the peaks skimage's peak_local_max finds (it is a speed-up only)."""
    import numpy as np
    from skimage.feature import peak_local_max
    from nuclei_stage1.segmentation import _anisotropic_seed_footprint, _fast_peak_coords
    rng = np.random.default_rng(0)
    for vs in [(2.0, 0.5616, 0.528), (1.0, 0.5, 0.5)]:
        for msd in [4, 12]:
            fp = _anisotropic_seed_footprint(msd, vs)
            for ties in (False, True):
                d = rng.random((24, 40, 44))
                d = np.round(d * 5) if ties else d
                m = rng.random(d.shape) > 0.3
                a = _fast_peak_coords(d, fp, m, 0.15)
                b = peak_local_max(d, footprint=fp, labels=m, exclude_border=False, threshold_rel=0.15)
                assert set(map(tuple, a)) == set(map(tuple, b))


def test_lazy_solidity_gives_same_decisions():
    """classify_instances skips the slow convex hull when area / bbox volume already clears the threshold;
    the accepted/difficult decisions must match the full explain_instances report exactly."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.measure import label
    from nuclei_stage1.segmentation import classify_instances, explain_instances, is_effectively_accepted
    rng = np.random.default_rng(3)
    for thr in (0.15, 0.55, 0.8):
        v = ndi.gaussian_filter(rng.random((30, 50, 50)), (2, 3, 3))
        lab = label(v > np.percentile(v, 60))
        acc, dif = classify_instances(lab, 15, thr, 4.0, accept_big_side_flags=True)
        a = np.zeros(lab.max() + 1, dtype=lab.dtype); d = np.zeros(lab.max() + 1, bool)
        for r in explain_instances(lab, 15, thr, 4.0):
            if is_effectively_accepted(r, True): a[r["label"]] = r["label"]
            else: d[r["label"]] = True
        assert np.array_equal(acc, a[lab]) and np.array_equal(dif, d[lab])

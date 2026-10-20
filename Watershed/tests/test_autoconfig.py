import json
from dataclasses import replace

import numpy as np

from helpers import add_ellipsoid, noisy
from nuclei_stage1 import Stage1Config, run_stage1
from nuclei_stage1.autoconfig import add_exemplar, load_exemplars, match_vector, select_config
from nuclei_stage1.config import PRESETS
from nuclei_stage1.descriptors import describe_chunk
from nuclei_stage1.segmentation import bandpass_mask

ISO = replace(Stage1Config(), voxel_size=(1.0, 1.0, 1.0), min_seed_distance=6, nucleus_diameter_vox=12.0)


def _tilted_volume(seed=0):
    """Nuclei on a strong slow illumination gradient: a raw percentile mask follows the gradient, a band-pass does not."""
    v = noisy((48, 64, 96), seed)
    v += np.linspace(0, 400, 96, dtype=np.float32)[None, None, :]        # slow ramp along x
    truth = np.zeros(v.shape, bool)
    for c in [(24, 16, 12), (24, 40, 30), (24, 16, 50), (24, 44, 70), (24, 20, 86)]:
        add_ellipsoid(v, c, (6, 6, 6), 1000.0 + 150)                      # same absolute brightness everywhere
        zz, yy, xx = np.indices(v.shape)
        truth |= (((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= 36)
    return v, truth


def test_bandpass_mask_is_off_by_default_and_changes_nothing():
    assert Stage1Config().mask_bandpass_um is None
    v, _ = _tilted_volume()
    a = run_stage1(v, ISO)[2]
    b = run_stage1(v, replace(ISO, mask_bandpass_um=None))[2]
    assert np.array_equal(a, b)


def test_bandpass_mask_ignores_a_slow_background_ramp():
    v, truth = _tilted_volume()
    pct = 100 * (1 - truth.mean() * 1.1)
    m = bandpass_mask(v, (1.0, 1.0, 1.0), 1.0, 8.0, pct)
    dice = 2 * (m & truth).sum() / (m.sum() + truth.sum())
    assert dice > 0.6
    flat = np.zeros(v.shape, bool)
    for z in range(v.shape[0]):
        flat[z] = v[z] > np.percentile(v[z], pct)
    assert dice > 2 * (flat & truth).sum() / (flat.sum() + truth.sum())    # beats the per-slice percentile here


def test_bandpass_respects_valid_mask():
    v, _ = _tilted_volume()
    valid = np.ones(v.shape, bool); valid[:, :, :10] = False
    m = bandpass_mask(v, (1.0, 1.0, 1.0), 1.0, 8.0, 94.0, valid=valid)
    assert not m[~valid].any()


def test_hippocampus_preset_is_applied_by_name():
    cfg = Stage1Config(**PRESETS["hippocampus"])
    assert cfg.mask_bandpass_um == (2.0, 8.0) and cfg.target_foreground_percentile == 94.0
    assert Stage1Config().min_seed_distance == 17          # PFC defaults untouched by the preset


def _exemplars():
    e = load_exemplars()
    assert {x["name"] for x in e} >= {"pfc_chunk1", "hippo_chunk3"}
    return e


def test_registered_examples_are_well_formed():
    for e in _exemplars():
        assert len(e["match_vec"]) == 3 and set(e["overrides"]) <= set(Stage1Config.__dataclass_fields__)


def test_select_config_picks_nearest_and_falls_back_when_far():
    v, _ = _tilted_volume()
    d = describe_chunk(v)
    near = {"name": "near", "match_vec": match_vector(d).tolist(), "overrides": {"min_seed_distance": 9}}
    far = {"name": "far", "match_vec": (match_vector(d) + 5).tolist(), "overrides": {"min_seed_distance": 3}}
    cfg, info = select_config(v, exemplars=[far, near], max_distance=0.75)
    assert info["source"] == "exemplar:near" and cfg.min_seed_distance == 9
    cfg, info = select_config(v, exemplars=[far], max_distance=0.75)
    assert info["source"] == "global-default" and cfg.min_seed_distance == Stage1Config().min_seed_distance
    cfg, info = select_config(v, exemplars=[], max_distance=0.75)
    assert info["source"] == "global-default"


def test_plain_example_switches_a_preset_bandpass_off():
    v, _ = _tilted_volume()
    plain = {"name": "plain", "match_vec": match_vector(describe_chunk(v)).tolist(), "overrides": {}}
    base = Stage1Config(**PRESETS["hippocampus"])
    cfg, info = select_config(v, exemplars=[plain], base=base)
    assert cfg.mask_bandpass_um is None


def test_json_lists_become_tuples_and_register_roundtrips(tmp_path):
    v, _ = _tilted_volume()
    p = tmp_path / "ex.json"
    add_exemplar(v, "t", {"mask_bandpass_um": (1.0, 8.0), "min_seed_distance": 12}, tuned_f1=0.7, path=p)
    assert json.loads(p.read_text())[0]["overrides"]["mask_bandpass_um"] == [1.0, 8.0]
    cfg, info = select_config(v, exemplars=json.loads(p.read_text()))
    assert cfg.mask_bandpass_um == (1.0, 8.0) and cfg.min_seed_distance == 12

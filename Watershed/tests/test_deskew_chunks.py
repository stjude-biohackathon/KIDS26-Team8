import numpy as np

from nuclei_stage1.chunks import list_chunks, parse_chunk_name
from nuclei_stage1.deskew import SLOPE, deskew_volume, to_deskewed_points, to_native_points


def test_points_round_trip():
    p = np.array([[10.0, 20.0, 30.0], [100.0, 5.0, 77.0]])
    back = to_native_points(to_deskewed_points(p, (128, 128, 128)), (128, 128, 128))
    assert np.allclose(back, p)


def test_deskew_widens_x_and_marks_padding():
    vol = np.ones((64, 16, 32), np.uint16)
    out, valid = deskew_volume(vol)
    assert out.shape[2] > vol.shape[2] and not valid.all() and valid.any()
    assert abs(SLOPE - 0.1228) < 1e-3


def test_chunk_names_keep_rois_apart(tmp_path):
    (tmp_path / "raw_chunked").mkdir()
    for n in ("roiA_z000000_y000000_x000000", "roiB_z000000_y000000_x000000", "junk"):
        (tmp_path / "raw_chunked" / f"{n}.tif").write_bytes(b"")
    cs = list_chunks(tmp_path)
    assert [c.roi_stem for c in cs] == ["roiA", "roiB"]
    assert parse_chunk_name("a_b_z000128_y000000_x000256") == ("a_b", 128, 0, 256)
    assert parse_chunk_name("nonsense") is None

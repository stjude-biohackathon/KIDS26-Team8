import subprocess
import sys
from pathlib import Path

from helpers import make_slab

REPO = Path(__file__).resolve().parent.parent


def test_pair_png_is_written(tmp_path):
    slab = make_slab(tmp_path / "s", [((60, 64, 60), (3, 9, 9)), ((60, 190, 70), (3, 9, 9))])
    for mod in ("nuclei_stage1.run_slab",):
        r = subprocess.run([sys.executable, "-m", mod, str(slab)], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "nuclei_stage1.save_slab_pair", str(slab), "--roi", "all",
                        "--downsample", "2", "--views", "proj_xy,sec_xy,proj_xz", "--width-px", "1200"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    png = slab / "figures" / f"{slab.name}_roiA_pair.png"
    assert png.exists() and png.stat().st_size > 10_000


def test_missing_labels_folder_is_a_loud_error(tmp_path):
    slab = make_slab(tmp_path / "s", [((60, 64, 60), (3, 9, 9))])      # raw tiles only, stage 1 never run
    r = subprocess.run([sys.executable, "-m", "nuclei_stage1.spin_slab_pair", str(slab), "--roi", "all",
                        "--downsample", "2", "--frames", "4", "--size", "160"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "no label files found" in (r.stdout + r.stderr)

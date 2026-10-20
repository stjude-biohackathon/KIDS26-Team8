import subprocess
import sys
from pathlib import Path

from helpers import make_slab

REPO = Path(__file__).resolve().parent.parent


def test_spin_gif_is_written(tmp_path):
    slab = make_slab(tmp_path / "s", [((60, 64, 60), (3, 9, 9)), ((60, 190, 70), (3, 9, 9))])
    r = subprocess.run([sys.executable, "-m", "nuclei_stage1.run_slab", str(slab)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "nuclei_stage1.spin_slab_pair", str(slab), "--roi", "all",
                        "--downsample", "2", "--frames", "4", "--size", "160"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    from PIL import Image
    gif = slab / "figures" / f"{slab.name}_roiA_spin.gif"
    assert gif.exists() and Image.open(gif).n_frames == 4

from pathlib import Path

import cc3d
import numpy as np
import tifffile


def load_volume(path):
    return tifffile.imread(str(path))


def load_ground_truth_as_labels(path):
    """Ilastik-derived ground truth is binary; derive instance labels via
    26-connectivity connected components (same tool/connectivity Peter's
    Unet/compare_pred_masks.py already uses)."""
    binary = tifffile.imread(str(path)) > 0
    return cc3d.connected_components(binary, connectivity=26)


def smallest_label_dtype(max_label):
    if max_label <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_label <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def write_label_mask(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = smallest_label_dtype(int(labels.max(initial=0)))
    tifffile.imwrite(str(path), labels.astype(dtype))

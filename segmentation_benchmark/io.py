from pathlib import Path

import numpy as np
import tifffile
from skimage import measure
from skimage.segmentation import relabel_sequential


TIFF_SUFFIXES = {".tif", ".tiff"}


def discover_volumes(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Volume directory does not exist: {root}")

    volumes = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    if not volumes:
        raise ValueError(f"No .tif or .tiff volumes found under {root}")
    return volumes


def matching_mask_path(mask_root, relative_path):
    candidate = Path(mask_root) / relative_path
    if candidate.is_file():
        return candidate

    alternate_suffix = ".tiff" if candidate.suffix.lower() == ".tif" else ".tif"
    alternate = candidate.with_suffix(alternate_suffix)
    if alternate.is_file():
        return alternate

    raise FileNotFoundError(
        f"No matching mask for {relative_path}; expected {candidate} or {alternate}"
    )


def load_volume(path, channel=0, channel_axis=0):
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim == 4:
        axis = channel_axis % volume.ndim
        if not 0 <= channel < volume.shape[axis]:
            raise ValueError(
                f"Channel {channel} is out of bounds for axis {axis} of "
                f"{path} with shape {volume.shape}"
            )
        volume = np.take(volume, channel, axis=axis)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D TIFF, or a 4-D TIFF with a channel axis, but "
            f"{path} has shape {volume.shape}"
        )
    if not np.issubdtype(volume.dtype, np.number):
        raise TypeError(f"Volume {path} has non-numeric dtype {volume.dtype}")
    return volume.astype(np.float32, copy=False)


def load_mask(path, mode="auto"):
    mask = np.asarray(tifffile.imread(path))
    mask = np.squeeze(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, but {path} has shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
        raise TypeError(f"Mask {path} has non-numeric dtype {mask.dtype}")
    if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
        raise ValueError(f"Mask {path} contains NaN or infinite values")
    return as_instance_labels(mask, mode=mode)


def as_instance_labels(mask, mode="auto"):
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {mask.shape}")
    if mode not in {"auto", "binary", "labels"}:
        raise ValueError(f"Unknown mask mode: {mode}")

    unique = np.unique(mask)
    is_binary = mask.dtype == np.bool_ or unique.size <= 2
    if mode == "binary" or (mode == "auto" and is_binary):
        labels = measure.label(mask > 0, connectivity=1)
    else:
        if np.issubdtype(mask.dtype, np.floating):
            rounded = np.rint(mask)
            if not np.allclose(mask, rounded):
                raise ValueError(
                    "Instance-label masks must contain integer values. "
                    "Threshold probability maps before benchmarking."
                )
            mask = rounded
        if np.any(mask < 0):
            raise ValueError("Instance-label masks cannot contain negative labels")
        labels, _, _ = relabel_sequential(mask.astype(np.int64, copy=False))

    return labels.astype(label_dtype(int(labels.max())), copy=False)


def label_dtype(max_label):
    if max_label <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def write_label_mask(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    labels = labels.astype(label_dtype(int(labels.max(initial=0))), copy=False)
    tifffile.imwrite(
        path,
        labels,
        compression="zlib",
        metadata={"axes": "ZYX"},
    )

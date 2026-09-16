import importlib
import inspect
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

from .io import as_instance_labels, load_mask, matching_mask_path


class OtsuMethod:
    name = "otsu"
    reports_runtime = True

    def __init__(self, min_size=0, dark_foreground=False):
        self.min_size = min_size
        self.dark_foreground = dark_foreground

    def predict(self, volume, relative_path):
        binary = _otsu_binary(volume, self.dark_foreground)
        if self.min_size > 0:
            binary = morphology.remove_small_objects(binary, min_size=self.min_size)
        return measure.label(binary, connectivity=1)


class OtsuWatershedMethod:
    name = "otsu-watershed"
    reports_runtime = True

    def __init__(self, min_size=0, min_distance=3, dark_foreground=False):
        self.min_size = min_size
        self.min_distance = min_distance
        self.dark_foreground = dark_foreground

    def predict(self, volume, relative_path):
        binary = _otsu_binary(volume, self.dark_foreground)
        if self.min_size > 0:
            binary = morphology.remove_small_objects(binary, min_size=self.min_size)
        if not np.any(binary):
            return np.zeros(binary.shape, dtype=np.uint16)

        distance = ndi.distance_transform_edt(binary)
        coordinates = peak_local_max(
            distance,
            labels=binary,
            min_distance=self.min_distance,
            exclude_border=False,
        )
        marker_mask = np.zeros(binary.shape, dtype=bool)
        marker_mask[tuple(coordinates.T)] = True
        markers = measure.label(marker_mask, connectivity=1)
        if markers.max() == 0:
            return measure.label(binary, connectivity=1)
        return segmentation.watershed(-distance, markers, mask=binary)


class PrecomputedMethod:
    reports_runtime = False

    def __init__(self, name, root, mask_mode="auto"):
        self.name = name
        self.root = Path(root)
        self.mask_mode = mask_mode
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Precomputed mask directory for {name} does not exist: {self.root}"
            )

    def predict(self, volume, relative_path):
        path = matching_mask_path(self.root, relative_path)
        return load_mask(path, mode=self.mask_mode)


class CellposeMethod:
    name = "cellpose"
    reports_runtime = True

    def __init__(self, model_name="nuclei", model_path=None, diameter=None, device="auto"):
        try:
            models = importlib.import_module("cellpose.models")
        except ImportError as exc:
            raise RuntimeError(
                "Cellpose is not installed. Install it with `pip install cellpose`."
            ) from exc

        gpu = _use_cuda(device)
        self.gpu = gpu
        self.torch = importlib.import_module("torch") if gpu else None
        if model_path:
            self.model = models.CellposeModel(
                gpu=gpu, pretrained_model=str(model_path)
            )
        else:
            parameters = inspect.signature(models.CellposeModel).parameters
            model_argument = (
                {"model_type": model_name}
                if "model_type" in parameters
                else {"pretrained_model": model_name}
            )
            self.model = models.CellposeModel(gpu=gpu, **model_argument)
        self.diameter = diameter

    def predict(self, volume, relative_path):
        result = self.model.eval(
            volume,
            channels=[0, 0],
            diameter=self.diameter,
            do_3D=True,
        )
        masks = result[0] if isinstance(result, tuple) else result
        return as_instance_labels(masks, mode="labels")

    def synchronize(self):
        if self.gpu:
            self.torch.cuda.synchronize()


class StarDistMethod:
    name = "stardist"
    reports_runtime = True

    def __init__(self, model_name="3D_demo", model_path=None):
        try:
            models = importlib.import_module("stardist.models")
        except ImportError as exc:
            raise RuntimeError(
                "StarDist is not installed. Install `stardist` and its TensorFlow "
                "backend before selecting this method."
            ) from exc

        if model_path:
            path = Path(model_path).resolve()
            self.model = models.StarDist3D(
                config=None,
                name=path.name,
                basedir=str(path.parent),
            )
        else:
            self.model = models.StarDist3D.from_pretrained(model_name)

    def predict(self, volume, relative_path):
        low, high = np.percentile(volume, (1.0, 99.8))
        normalized = np.clip((volume - low) / max(high - low, 1e-8), 0.0, 1.0)
        labels, _ = self.model.predict_instances(normalized)
        return as_instance_labels(labels, mode="labels")


class UNetMethod:
    name = "unet"
    reports_runtime = True

    def __init__(
        self,
        model_path,
        num_layers=3,
        chunk_size=850,
        patch_size=128,
        threshold=0.5,
        device="auto",
    ):
        if not model_path:
            raise ValueError("The unet method requires --unet-model")
        try:
            torch = importlib.import_module("torch")
            unet_module = importlib.import_module("Unet.unet_class")
        except ImportError as exc:
            raise RuntimeError(
                "The unet method requires PyTorch and the repository's Unet package."
            ) from exc
        self.torch = torch
        self.device = torch.device(
            "cuda" if _use_cuda(device) else "cpu"
        )
        self.model = unet_module.UNet3D(num_layers=num_layers).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.chunk_size = chunk_size
        self.patch_size = (patch_size,) * 3
        self.threshold = threshold

    def predict(self, volume, relative_path):
        binary = np.zeros(volume.shape, dtype=bool)
        for z in range(0, volume.shape[0], self.chunk_size):
            for y in range(0, volume.shape[1], self.chunk_size):
                for x in range(0, volume.shape[2], self.chunk_size):
                    slices = (
                        slice(z, min(z + self.chunk_size, volume.shape[0])),
                        slice(y, min(y + self.chunk_size, volume.shape[1])),
                        slice(x, min(x + self.chunk_size, volume.shape[2])),
                    )
                    prediction = self._predict_chunk(volume[slices])
                    binary[slices] = prediction > 0
        return measure.label(binary, connectivity=1)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _predict_chunk(self, chunk):
        padding = [
            (size - (chunk.shape[index] % size)) % size
            for index, size in enumerate(self.patch_size)
        ]
        padded = np.pad(
            chunk,
            [(0, amount) for amount in padding],
            mode="constant",
        )
        output = np.zeros(padded.shape, dtype=bool)

        with self.torch.inference_mode():
            for z in range(0, padded.shape[0], self.patch_size[0]):
                for y in range(0, padded.shape[1], self.patch_size[1]):
                    for x in range(0, padded.shape[2], self.patch_size[2]):
                        slices = (
                            slice(z, z + self.patch_size[0]),
                            slice(y, y + self.patch_size[1]),
                            slice(x, x + self.patch_size[2]),
                        )
                        patch = self.torch.from_numpy(
                            padded[slices].astype(np.float32, copy=False)
                        )
                        patch = patch.unsqueeze(0).unsqueeze(0).to(self.device)
                        probability = self.torch.sigmoid(self.model(patch))
                        output[slices] = (
                            probability.squeeze().detach().cpu().numpy()
                            > self.threshold
                        )

        return output[
            : chunk.shape[0],
            : chunk.shape[1],
            : chunk.shape[2],
        ]


def _otsu_binary(volume, dark_foreground):
    finite = np.asarray(volume)[np.isfinite(volume)]
    if finite.size != volume.size:
        raise ValueError("Input volume contains NaN or infinite values")
    if finite.size == 0:
        raise ValueError("Input volume is empty")
    if np.all(finite == finite.flat[0]):
        return np.zeros(volume.shape, dtype=bool)
    threshold = filters.threshold_otsu(finite)
    return volume < threshold if dark_foreground else volume > threshold


def _use_cuda(device):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device: {device}")
    if device == "cpu":
        return False
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if device == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch is not installed")
        return False
    available = bool(torch.cuda.is_available())
    if device == "cuda" and not available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return available

#!/usr/bin/env python3
"""
predict_volume.py — run UNet3D nuclei detection on a full volume from the terminal.

Accepts a tiff stack OR an omehans directory as input.
Iterates over chunks internally (no chunk number needed).
Writes a single assembled mask tiff and a centroids CSV.

If your volume has a different voxel size than the model was trained on,
pass --voxel-size and --model-voxel-size. The volume is resampled to match
the training resolution before inference, and centroids are mapped back to
the original voxel coordinates in the output CSV.

Usage
-----
# tiff stack:
python predict_volume.py --model /path/to/model.pth --input /path/to/vol.tif --out-dir ./detections

# omehans directory:
python predict_volume.py --model /path/to/model.pth --input /path/to/vol.omehans --out-dir ./detections

# higher-resolution input (voxel size 0.5 µm isotropic, model trained at 1.54 µm):
python predict_volume.py --model model.pth --input hires.tif --out-dir ./out \\
    --voxel-size 0.5 0.5 0.5 --model-voxel-size 1.34 1.54 2.0

# override chunk size for a small test volume (default 850):
python predict_volume.py --model /path/to/model.pth --input vol.tif --out-dir ./out --chunk-size 256

# select channel (default 0 = nuclei):
python predict_volume.py --model /path/to/model.pth --input vol.omehans --out-dir ./out --channel 1
"""

import argparse
import os
import sys

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as G
import pandas as pd
from skimage import measure
from skimage.transform import resize, rescale


# ── Model ─────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1);  self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1);          self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, 3, padding=1);         self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, 3, padding=1);         self.bn4 = nn.BatchNorm3d(512)
        self.conv5 = nn.Conv3d(512, 1024, 3, padding=1);        self.bn5 = nn.BatchNorm3d(1024)
        self.upconv6 = nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.conv6 = nn.Conv3d(1024, 512, 3, padding=1);        self.bn6 = nn.BatchNorm3d(512)
        self.upconv7 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.conv7 = nn.Conv3d(512, 256, 3, padding=1);         self.bn7 = nn.BatchNorm3d(256)
        self.upconv8 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv8 = nn.Conv3d(256, 128, 3, padding=1);         self.bn8 = nn.BatchNorm3d(128)
        self.upconv9 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv3d(128, 64, 3, padding=1);          self.bn9 = nn.BatchNorm3d(64)
        self.output = nn.Conv3d(64, out_channels, 1)

    def forward(self, x):
        c1 = G.relu(self.bn1(self.conv1(x)))
        c2 = G.relu(self.bn2(self.conv2(G.max_pool3d(c1, 2, 2))))
        c3 = G.relu(self.bn3(self.conv3(G.max_pool3d(c2, 2, 2))))
        c4 = G.relu(self.bn4(self.conv4(G.max_pool3d(c3, 2, 2))))
        c7 = G.relu(self.bn7(self.conv7(torch.cat([self.upconv7(c4), c3], 1))))
        c8 = G.relu(self.bn8(self.conv8(torch.cat([self.upconv8(c7), c2], 1))))
        c9 = G.relu(self.bn9(self.conv9(torch.cat([self.upconv9(c8), c1], 1))))
        return self.output(c9)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_volume(path, channel):
    """Return a 3-D numpy array (Z, Y, X) from a tiff or omehans path."""
    if path.endswith('.tif') or path.endswith('.tiff'):
        vol = tifffile.imread(path).astype(np.float32)
        if vol.ndim == 4:           # (C, Z, Y, X)
            vol = vol[channel]
        elif vol.ndim == 3:
            pass                    # already (Z, Y, X)
        else:
            raise ValueError(f"Unexpected tiff ndim={vol.ndim}")
        return vol

    # omehans / zarr
    import dask.array as da
    import zarr
    try:
        from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
        store = H5_Nested_Store(path)
    except Exception:
        from stack_to_multiscale_ngff.h5_nested_store import H5_Nested_Store
        store = H5_Nested_Store(path)
    z = zarr.open(store)
    arr = da.from_zarr(z)
    print(f"omehans shape: {arr.shape}")
    # normalise to (Z, Y, X)
    if arr.ndim == 5:               # (T, C, Z, Y, X)
        arr = arr[0, channel]
    elif arr.ndim == 4:             # (C, Z, Y, X) or (T, Z, Y, X) — assume first axis is C
        arr = arr[channel]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected omehans ndim={arr.ndim}")
    return arr.compute().astype(np.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

def normalize(arr):
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-8)


def predict_chunk(model, chunk, patch_size, threshold, device):
    """Run sliding-window 3-D UNet on one chunk; return float32 segmentation array."""
    vol_size = chunk.shape
    pad = [
        (patch_size[d] - ((vol_size[d] - patch_size[d]) % patch_size[d])) % patch_size[d]
        for d in range(3)
    ]
    padded = np.pad(chunk, [(0, pad[d]) for d in range(3)], mode='linear_ramp')
    out = np.zeros(padded.shape, dtype=np.float32)

    for i in range(0, padded.shape[0], patch_size[0]):
        for j in range(0, padded.shape[1], patch_size[1]):
            for k in range(0, padded.shape[2], patch_size[2]):
                patch = padded[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                #patch = normalize(patch)
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                with torch.no_grad():
                    pred = (torch.sigmoid(model(t)) > threshold).float()
                    pred = pred.squeeze().cpu().numpy()
                    pred = np.interp(pred, (pred.min(), pred.max()), (0, 255))
                    if np.all(pred > 0):
                        pred = np.zeros_like(pred)
                out[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]] += pred

    return out[:vol_size[0], :vol_size[1], :vol_size[2]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run UNet3D nuclei detection on a full volume (tiff or omehans).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model',      required=True, help='Path to model .pth file')
    parser.add_argument('--input',      required=True, help='Input volume: .tif/.tiff or omehans directory')
    parser.add_argument('--out-dir',    required=True, help='Output directory for mask.tif and centroids.csv')
    parser.add_argument('--channel',    type=int, default=0,
                        help='Channel to use if input is multi-channel (default: 0 = nuclei)')
    parser.add_argument('--chunk-size', type=int, default=850,
                        help='Chunk size in voxels per side (default: 850; use smaller for test volumes)')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='UNet patch size in voxels per side (default: 128)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Sigmoid threshold for binarisation (default: 0.5)')
    parser.add_argument('--voxel-size', type=float, nargs=3, metavar=('Z', 'Y', 'X'),
                        default=None,
                        help='Voxel size of the input volume in µm (Z Y X). '
                             'Required together with --model-voxel-size to trigger resampling.')
    parser.add_argument('--model-voxel-size', type=float, nargs=3, metavar=('Z', 'Y', 'X'),
                        default=[1, 1, 1],
                        help='Voxel size the model was trained on in µm (Z Y X). '
                             'Default: 1.34 1.54 2.0 (NPBB328 nuclei)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load volume
    print(f'Loading {args.input} ...')
    vol = load_volume(args.input, args.channel)
    original_shape = vol.shape
    print(f'Volume shape: {vol.shape}  dtype: {vol.dtype}')

    # Resample to model resolution if voxel sizes differ
    scale_factors = None
    if args.voxel_size is not None:
        scale_factors = np.array(args.voxel_size) / np.array(args.model_voxel_size)
        if not np.allclose(scale_factors, 1.0, atol=0.01):
            new_shape = tuple(int(round(s * f)) for s, f in zip(vol.shape, scale_factors))
            print(f'Resampling {vol.shape} → {new_shape}  '
                  f'(voxel {args.voxel_size} µm → {args.model_voxel_size} µm)')
            vol = resize(vol, new_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
            print(f'Resampled shape: {vol.shape}')
        else:
            print('Voxel sizes match model training resolution — no resampling needed.')
            scale_factors = None

    # Load model
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f'Model loaded from {args.model}')

    # Build chunk grid
    chunk_size = (args.chunk_size,) * 3
    patch_size = (args.patch_size,) * 3

    ratios = (np.array(vol.shape) / np.array(chunk_size)).astype(int) + 1
    origins = []
    for z in range(ratios[0]):
        for y in range(ratios[1]):
            for x in range(ratios[2]):
                origins.append((z * chunk_size[0], y * chunk_size[1], x * chunk_size[2]))

    n_chunks = len(origins)
    print(f'Processing {n_chunks} chunk(s) of size {chunk_size}')

    # Assemble output mask in full-volume space
    mask = np.zeros(vol.shape, dtype=np.uint8)

    for ci, (oz, oy, ox) in enumerate(origins):
        sz = slice(oz, min(oz + chunk_size[0], vol.shape[0]))
        sy = slice(oy, min(oy + chunk_size[1], vol.shape[1]))
        sx = slice(ox, min(ox + chunk_size[2], vol.shape[2]))
        chunk = vol[sz, sy, sx]
        print(f'  Chunk {ci+1}/{n_chunks}  shape={chunk.shape}  origin=({oz},{oy},{ox})')

        seg = predict_chunk(model, chunk, patch_size, args.threshold, device)
        seg_bin = (seg > 0).astype(np.uint8) * 255
        mask[sz, sy, sx] = np.maximum(mask[sz, sy, sx], seg_bin)

    # Write mask
    mask_path = os.path.join(args.out_dir, 'mask.tif')
    tifffile.imwrite(mask_path, mask)
    print(f'Mask saved → {mask_path}')

    # Centroids
    labels = measure.label(mask)
    if labels.max() > 0:
        table = pd.DataFrame(measure.regionprops_table(labels, properties=['centroid']))
        table.columns = ['axis-0', 'axis-1', 'axis-2']

        # Map centroid coordinates back to original (pre-resample) voxel space
        if scale_factors is not None:
            inv = 1.0 / scale_factors
            for i, col in enumerate(['axis-0', 'axis-1', 'axis-2']):
                table[col] = table[col] * inv[i]
            print(f'Centroids remapped to original voxel coordinates (scale {inv.round(3)})')

        centroids_path = os.path.join(args.out_dir, 'centroids.csv')
        table.to_csv(centroids_path, index=False)
        print(f'Centroids ({len(table)} nuclei) saved → {centroids_path}')
    else:
        print('No nuclei detected.')


if __name__ == '__main__':
    main()
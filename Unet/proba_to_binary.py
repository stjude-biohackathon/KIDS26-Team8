import tifffile as tiff
import numpy as np
import h5py
import os

# Read probability map, threshold and save as a binary tif

with h5py.File('/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/probabilities/chunk_z5_y36_x0_Probabilities.h5', 'r') as f:
    proba_map = f['exported_data'][:]  # shape will be (Z, Y, X, 2)

# take foreground channel (channel 0)
prob_foreground = proba_map[..., 0]
print(prob_foreground.dtype, prob_foreground.min(), prob_foreground.max())

binary = (prob_foreground > 0.5).astype(np.uint8) * 255

os.makedirs('/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/masks', exist_ok=True)
tiff.imwrite('/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/masks/chunk_z5_y36_x0.tif', binary)
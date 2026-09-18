"""View a raw volume alongside predicted/ground-truth masks in napari.

Run from a napari-enabled Python environment (e.g. in an OnDemand remote
desktop session), not from the headless HPC login shell.

Usage:
    python view_chunk.py --raw RAW.tif [--predicted PRED.tif] \
        [--ground-truth GT.tif] [--gt-mode auto|binary|labels]

--gt-mode auto (default) treats the ground-truth tif as binary and derives
instance labels via 26-connectivity connected components, matching exactly
what pretrained_baselines/evaluate.py scores against. Pass --gt-mode labels
if the file is already instance-labeled, or --gt-mode binary to view it as
a plain binary mask without relabeling.
"""

import argparse

import napari
import numpy as np
import tifffile
from skimage.measure import label


def load_ground_truth(path, mode):
    arr = tifffile.imread(path)
    if mode == "labels":
        return arr
    binary = arr > 0
    if mode == "binary":
        return binary
    return label(binary, connectivity=3)  # matches cc3d connectivity=26


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--predicted")
    parser.add_argument("--ground-truth")
    parser.add_argument("--gt-mode", choices=["auto", "binary", "labels"], default="auto")
    args = parser.parse_args()

    viewer = napari.Viewer()
    viewer.add_image(tifffile.imread(args.raw), name="raw", colormap="gray")

    if args.predicted:
        viewer.add_labels(tifffile.imread(args.predicted), name="predicted")

    if args.ground_truth:
        gt = load_ground_truth(args.ground_truth, args.gt_mode)
        viewer.add_labels(gt, name="ground_truth (ilastik)")

    napari.run()


if __name__ == "__main__":
    main()

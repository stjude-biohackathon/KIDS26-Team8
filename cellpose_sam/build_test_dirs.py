"""Build symlink farms for evaluation, from the shared hackathon data.

Tier 1: split_manifest.json's designated test-split chunks (pfCortex only --
Cerebellum's masks_chunked_v2 covers the 384 TRAIN chunks, not test, and
masks_chunked (v1) is essentially unpopulated for Cerebellum; hippocampus
isn't chunked at all, no manifest entry for it).

Tier 1 (non-standard, Cerebellum only): the 384 chunks masks_chunked_v2
actually covers are the TRAIN split, not the official held-out test set.
Since Cellpose-SAM/uSAM are pretrained/inference-only (never trained on any
of this data), there's no leakage risk in scoring against these -- but they
are NOT the same evaluation set the U-Net will be scored on, so this is kept
in a separately-named tier1_trainsplit/ directory and must be labeled as
such in any results/write-up, not silently compared to the official test
numbers.

Tier 2: whole, unchunked slab volumes (all three regions) -- same shape
(256x1024x1280) across regions, used for cross-region comparison and
larger-than-chunk runtime/memory testing. Cerebellum has no whole-volume
ground truth, so its ground_truth/ dir is left empty for that region.
"""

import json
from pathlib import Path

DATA_ROOT = Path("/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026")
MANIFEST_PATH = DATA_ROOT / "split_manifest.json"
OUT_ROOT = Path.home() / "cellpose_baseline" / "test_data"

TIER1_REGIONS = {
    # Official held-out test split. masks_chunked_v2 covers only the 384 TRAIN
    # chunks for both regions (verified: 0/96 test-chunk overlap) -- it's
    # train-only re-annotated GT, not a general upgrade over v1. masks_chunked
    # (v1) has full 96/96 test coverage for pfCortex. Cerebellum's masks_chunked
    # has only 1 file total -- no usable test-split GT exists yet, so it's left
    # out here (see TIER1_TRAINSPLIT_REGIONS for the non-standard alternative).
    "slab06_Cerebellum_chunks": None,
    "slab21_pfCortex_chunks": "masks_chunked",
}

# Non-standard: the 384 TRAIN-split chunks, used only where the official test
# split has no usable ground truth. Safe for us specifically (inference-only,
# no training) but NOT comparable to methods scored on the official test set.
TIER1_TRAINSPLIT_REGIONS = {
    "slab06_Cerebellum_chunks": "masks_chunked_v2",
}

TIER2_REGIONS = {
    "slab06_Cerebellum_chunks": None,  # no whole-volume GT available
    "slab21_pfCortex_chunks": "masks",
    "slab19_hippocampus_chunks": "masks",
}


def symlink_force(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def build_tier1():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for region, gt_subdir in TIER1_REGIONS.items():
        region_dir = DATA_ROOT / region
        raw_src = region_dir / "raw_chunked"
        gt_src = region_dir / gt_subdir if gt_subdir is not None else None
        test_ids = manifest[region]["test"]
        gt_linked = 0
        for chunk_id in test_ids:
            fname = f"{chunk_id}.tif"
            symlink_force(raw_src / fname, OUT_ROOT / region / "tier1" / "raw" / fname)
            if gt_src is not None and (gt_src / fname).exists():
                symlink_force(gt_src / fname, OUT_ROOT / region / "tier1" / "ground_truth" / fname)
                gt_linked += 1
        print(f"tier1 {region}: linked {len(test_ids)} raw, {gt_linked}/{len(test_ids)} with ground truth")


def build_tier1_trainsplit():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for region, gt_subdir in TIER1_TRAINSPLIT_REGIONS.items():
        region_dir = DATA_ROOT / region
        raw_src = region_dir / "raw_chunked"
        gt_src = region_dir / gt_subdir
        train_ids = manifest[region]["train"]
        gt_linked = 0
        for chunk_id in train_ids:
            fname = f"{chunk_id}.tif"
            symlink_force(raw_src / fname, OUT_ROOT / region / "tier1_trainsplit" / "raw" / fname)
            if (gt_src / fname).exists():
                symlink_force(gt_src / fname, OUT_ROOT / region / "tier1_trainsplit" / "ground_truth" / fname)
                gt_linked += 1
        print(f"tier1_trainsplit {region}: linked {len(train_ids)} raw, {gt_linked}/{len(train_ids)} with ground truth")


def build_tier2():
    for region, gt_subdir in TIER2_REGIONS.items():
        region_dir = DATA_ROOT / region
        raw_files = sorted((region_dir / "raw").glob("*.tif"))
        for f in raw_files:
            symlink_force(f, OUT_ROOT / region / "tier2" / "raw" / f.name)
        if gt_subdir is not None:
            gt_files = sorted((region_dir / gt_subdir).glob("*.tif"))
            for f in gt_files:
                symlink_force(f, OUT_ROOT / region / "tier2" / "ground_truth" / f.name)
            print(f"tier2 {region}: linked {len(raw_files)} raw, {len(gt_files)} ground truth")
        else:
            print(f"tier2 {region}: linked {len(raw_files)} raw, no ground truth available")


if __name__ == "__main__":
    build_tier1()
    build_tier1_trainsplit()
    build_tier2()

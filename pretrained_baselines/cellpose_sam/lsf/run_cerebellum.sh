#!/bin/bash
#BSUB -q biohackathon
#BSUB -J cellpose_sam_cerebellum
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 6:00
#BSUB -o /home/efoste34/cellpose_baseline/logs/cerebellum_%J.out
#BSUB -e /home/efoste34/cellpose_baseline/logs/cerebellum_%J.err

set -euo pipefail
ENV_PY=/research/rgs01/home/clusterHome/efoste34/.conda/envs/cellpose_env/bin/python3
REPO=/research/rgs01/home/clusterHome/efoste34/KIDS26-Team8
BASE=/home/efoste34/cellpose_baseline
REGION=slab06_Cerebellum_chunks

# Tier 1 (non-standard split): Cerebellum's official 96-chunk test split has
# no usable ground truth (masks_chunked_v2 covers only the 384 TRAIN chunks;
# masks_chunked/v1 is essentially unpopulated). Since Cellpose-SAM never
# trains on this data, scoring against the 384 train-labeled chunks carries
# no leakage risk for us -- but these numbers are NOT directly comparable to
# methods (e.g. the U-Net) scored on the official test split. Output is
# named _tier1_trainsplit to keep that distinction visible everywhere.
"$ENV_PY" "$REPO/pretrained_baselines/cellpose_sam/run_inference.py" \
  --raw-dir "$BASE/test_data/$REGION/tier1_trainsplit/raw" \
  --masks-out-dir "$BASE/runs/cellpose_sam/$REGION/tier1_trainsplit/masks" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier1_trainsplit/runtime.csv" \
  --device cuda

"$ENV_PY" "$REPO/pretrained_baselines/evaluate.py" \
  --masks-dir "$BASE/runs/cellpose_sam/$REGION/tier1_trainsplit/masks" \
  --ground-truth-dir "$BASE/test_data/$REGION/tier1_trainsplit/ground_truth" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier1_trainsplit/runtime.csv" \
  --output-csv "$REPO/pretrained_baselines/results/cellpose_sam/${REGION}_tier1_trainsplit.csv"

# Tier 2: 3 whole slab volumes -- no ground truth available for Cerebellum,
# so this is runtime/memory-scaling data only, no evaluate.py step.
"$ENV_PY" "$REPO/pretrained_baselines/cellpose_sam/run_inference.py" \
  --raw-dir "$BASE/test_data/$REGION/tier2/raw" \
  --masks-out-dir "$BASE/runs/cellpose_sam/$REGION/tier2/masks" \
  --runtime-csv "$REPO/pretrained_baselines/results/cellpose_sam/${REGION}_tier2_runtime_only.csv" \
  --device cuda

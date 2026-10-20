#!/bin/bash
#BSUB -q biohackathon
#BSUB -J cellpose_sam_pfcortex
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 6:00
#BSUB -o /home/efoste34/cellpose_baseline/logs/pfcortex_%J.out
#BSUB -e /home/efoste34/cellpose_baseline/logs/pfcortex_%J.err

set -euo pipefail
ENV_PY=/research/rgs01/home/clusterHome/efoste34/.conda/envs/cellpose_env/bin/python3
REPO=/research/rgs01/home/clusterHome/efoste34/KIDS26-Team8
BASE=/home/efoste34/cellpose_baseline
REGION=slab21_pfCortex_chunks

# Tier 1: 96 held-out test chunks (128^3)
"$ENV_PY" "$REPO/pretrained_baselines/cellpose_sam/run_inference.py" \
  --raw-dir "$BASE/test_data/$REGION/tier1/raw" \
  --masks-out-dir "$BASE/runs/cellpose_sam/$REGION/tier1/masks" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier1/runtime.csv" \
  --device cuda

"$ENV_PY" "$REPO/pretrained_baselines/evaluate.py" \
  --masks-dir "$BASE/runs/cellpose_sam/$REGION/tier1/masks" \
  --ground-truth-dir "$BASE/test_data/$REGION/tier1/ground_truth" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier1/runtime.csv" \
  --output-csv "$REPO/pretrained_baselines/results/cellpose_sam/${REGION}_tier1.csv"

# Tier 2: 3 whole slab volumes (256x1024x1280)
"$ENV_PY" "$REPO/pretrained_baselines/cellpose_sam/run_inference.py" \
  --raw-dir "$BASE/test_data/$REGION/tier2/raw" \
  --masks-out-dir "$BASE/runs/cellpose_sam/$REGION/tier2/masks" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier2/runtime.csv" \
  --device cuda

"$ENV_PY" "$REPO/pretrained_baselines/evaluate.py" \
  --masks-dir "$BASE/runs/cellpose_sam/$REGION/tier2/masks" \
  --ground-truth-dir "$BASE/test_data/$REGION/tier2/ground_truth" \
  --runtime-csv "$BASE/runs/cellpose_sam/$REGION/tier2/runtime.csv" \
  --output-csv "$REPO/pretrained_baselines/results/cellpose_sam/${REGION}_tier2.csv"

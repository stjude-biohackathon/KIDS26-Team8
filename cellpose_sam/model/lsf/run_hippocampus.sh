#!/bin/bash
#BSUB -q biohackathon
#BSUB -J cellpose_sam_hippocampus
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 6:00
#BSUB -o /home/efoste34/cellpose_baseline/logs/hippocampus_%J.out
#BSUB -e /home/efoste34/cellpose_baseline/logs/hippocampus_%J.err

set -euo pipefail
ENV_PY=/research/rgs01/home/clusterHome/efoste34/.conda/envs/cellpose_env/bin/python3
REPO=/research/rgs01/home/clusterHome/efoste34/KIDS26-Team8
BASE=/home/efoste34/cellpose_baseline
REGION=slab19_hippocampus_chunks

# Hippocampus is not chunked -- tier 2 (3 whole volumes) is its only data.
# Small n=3, no train/test split -- informal, caveat this in any write-up.
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

#!/bin/bash
set -e

ENV_IDS=("PickCube-v1" "PegInsertionSide-v1" "PushCube-v1" "PlugCharger-v1" "StackCube-v1")

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <seed1> <seed2> ..."
    exit 1
fi
SEEDS=("$@")

NUM_TRAJ=25
MAX_STEP=300
CLEAN_MODELS=""
LIVE_VIEW=" --live-view"

# -----------------------------
# 반복 실행
# -----------------------------
for ENV_ID in "${ENV_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        if ! [[ $SEED =~ ^[0-9]+$ ]]; then
            echo "Skipping invalid seed: $SEED (must be numeric)"
            continue
        fi

        echo "============================================="
        echo "Running evaluation for ENV: $ENV_ID, SEED: $SEED"
        echo "============================================="

        ./evaluation.sh "$ENV_ID" "$SEED" --num_traj "$NUM_TRAJ" --max_step "$MAX_STEP" $CLEAN_MODELS $LIVE_VIEW
    done
done
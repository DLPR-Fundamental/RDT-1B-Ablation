#!/bin/bash
set -e

print_usage() {
    echo "Usage: ./evaluation.sh <env_id> <random_seed> --num_traj <num_traj> --max_step <max_step> --clean_models --live-view"
    echo "Arguments: "
    echo "  <env_id>      : <PegInsertionSide-v1|PickCube-v1|PlugCharger-v1|PushCube-v1|StackCube-v1>"
    echo "  <random_seed> : <random_seed> "
    echo "Options:"
    echo "  [--num_traj 25] [--max_step 300] [--clean_models] [--live-view]" 
    echo ""
    echo "Example: ./evaluation.sh PickCube-v1 10 --num_traj 25 --max_step 300 --clean_models --live-view"
}

if [ "$#" -lt 2 ]; then
    print_usage
    exit 1
fi

ENV_ID=$1
RANDOM_SEED=$2

if ! [[ $RANDOM_SEED =~ ^[0-9]+$ ]]; then
    echo "Error: <random_seed> must be a numeric value."
    exit 1
fi

NUM_TRAJ=25
MAX_STEP=300
CLEAN_MODELS="false"
IS_LIVE_VIEW=""

while [[ "$#" -gt 2 ]]; do
    case $3 in
        --num_traj) NUM_TRAJ="$4"; shift 2 ;;
        --max_step) MAX_STEP="$4"; shift 2 ;;
        --clean_models) CLEAN_MODELS="true"; shift ;;
        --live-view) IS_LIVE_VIEW="$3"; shift ;;
        *) echo "Unknown option passed: $3"; print_usage; exit 1 ;;
    esac
done

if [ "$CLEAN_MODELS" = "true" ]; then
    if [ -d "pretrained_models" ]; then
        echo "[INFO] Cleaning existing pretrained models..."
        rm -rf pretrained_models/*
    fi
fi

# Download lang_embeds
python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models lang_embeds/
export PYTHONPATH=$(pwd)/src/evaluation:$(pwd)/src/model/policy:$PYTHONPATH

# Evaluate on ManiSkill tasks with live view
############################## Diffusion Policy Evaluation ##############################
# Download
if [ ! -d "pretrained_models/diffusion_policy" ]; then
    echo "[INFO] diffusion_policy pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models diffusion_policy/
fi
# Evaluate
python -m src.evaluation.run --model dp --random_seed $RANDOM_SEED --env $ENV_ID --pretrained_path pretrained_models/diffusion_policy/700.ckpt --num-traj $NUM_TRAJ --max-step $MAX_STEP $IS_LIVE_VIEW

############################## RDT-1B Evaluation ##############################
# Download
if [ ! -d "pretrained_models/rdt" ]; then
    echo "[INFO] rdt pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/
fi
# Evaluate
LANG_EMBEDS_PATH=pretrained_models/lang_embeds/text_embed_${ENV_ID}.pt # Set language embeddings path
python -m src.evaluation.run --model rdt --random_seed $RANDOM_SEED --env $ENV_ID --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --num-traj $NUM_TRAJ  --max-step $MAX_STEP \
                            --action-downsample 4 --lang-embeddings-path $LANG_EMBEDS_PATH $IS_LIVE_VIEW




# # Download pretrained octo model(if not already present)
# if [ ! -d "pretrained_models/octo" ]; then
#     echo "[INFO] octo pretrained model not found. Downloading..."
#     python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models octo/
# fi
# export PYTHONPATH=$(pwd)/src/evaluation/octo
# python -m src.evaluation.eval_octo --env-id $ENV_ID --pretrained_path pretrained_models/octo/experiment_20241208_112612

# # Download pretrained openvla model(if not already present)
# if [ ! -d "pretrained_models/openvla" ]; then
#     echo "[INFO] openvla pretrained model not found. Downloading..."
#     python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models "openvla-7b*/"
#     mv pretrained_models/openvla-7b* pretrained_models/openvla
# fi
# python -m src.evaluation.eval_openvla --env-id $ENV_ID --pretrained_path pretrained_models/openvla


# # Old eval scripts
# python -m src.evaluation._old.eval_rdt --env-id $ENV_ID --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --lang_embeddings_path $LANG_EMBEDS_PATH --live-view
# python -m src.evaluation._old.eval_dp --env-id $ENV_ID --pretrained_path pretrained_models/diffusion_policy/700.ckpt --vis --save-video

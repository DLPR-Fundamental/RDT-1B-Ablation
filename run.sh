#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh <env_id> <clean_models>"
    echo "Example: ./run.sh <PegInsertionSide-v1|PickCube-v1|PlugCharger-v1|PushCube-v1|StackCube-v1> <true|false>"
    exit 1
fi

ENV_ID=$1
CLEAN_MODELS=$2

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
python -m src.evaluation.run --model dp --env $ENV_ID --pretrained_path pretrained_models/diffusion_policy/700.ckpt --num-traj 25 --live-view

############################## RDT-1B Evaluation ##############################
# Download
if [ ! -d "pretrained_models/rdt" ]; then
    echo "[INFO] rdt pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/
fi
# Evaluate
LANG_EMBEDS_PATH=pretrained_models/lang_embeds/text_embed_${ENV_ID}.pt # Set language embeddings path
python -m src.evaluation.run --model rdt --env PickCube-v1 --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --num-traj 25 --action-downsample 16 \
                             --lang-embeddings-path $LANG_EMBEDS_PATH --live-view




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

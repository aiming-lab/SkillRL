#!/bin/bash
set -x

# Paths — adjust if your layout differs
LLAMAFACTORY_DIR="/inspire/hdd/project/multi-agent/czxs253130170/LLaMA-Factory"
PYTHON="/inspire/hdd/project/multi-agent/czxs253130170/SHINE/shine_env/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALFWORLD_DATA="/inspire/hdd/project/multi-agent/czxs253130170/SHINE/alfworld_data/alfworld"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# dataset_info.json in LLaMA-Factory must contain 'skillrl_alfworld_sft' entry pointing to:
#   /inspire/hdd/project/multi-agent/czxs253130170/SkillRL/SkillRL-SFT-Data/alfworld/alfworld.jsonl

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    "$LLAMAFACTORY_DIR/src/llamafactory/launcher.py" \
    "$SCRIPT_DIR/alfworld.yaml" \
    "$@"

#!/bin/bash
set -e

LLAMAFACTORY_DIR="/inspire/hdd/project/multi-agent/czxs253130170/LLaMA-Factory"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LLAMAFACTORY_DIR/logs/sft"
LOG_FILE="$LOG_DIR/search_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Log: $LOG_FILE"
echo "Starting Search SFT at $(date)" | tee "$LOG_FILE"

cd "$LLAMAFACTORY_DIR"

cp "$SCRIPT_DIR/search.yaml" examples/train_full/qwen3_8b_skillrl_search_full.yaml

nohup llamafactory-cli train examples/train_full/qwen3_8b_skillrl_search_full.yaml \
    "$@" \
    2>&1 | tee -a "$LOG_FILE" &

PID=$!
echo $PID > "$LOG_DIR/search_sft.pid"
echo "Launched PID=$PID — tail log with:"
echo "  tail -f $LOG_FILE"

#!/bin/bash
set -o pipefail

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p nohup_log
LOGFILE=nohup_log/s_offtype_SCD_QS2_$(date +%Y%m%d_%H%M%S).log

TRAIN_EXIT_CODE=1

cleanup() {
    echo "TRAIN_EXIT_CODE=$TRAIN_EXIT_CODE" | tee -a "$LOGFILE"

    if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
        echo "OK -> shutdown" | tee -a "$LOGFILE"
        /usr/bin/shutdown
    else
        echo "FAILED -> NOT shutdown" | tee -a "$LOGFILE"
    fi
}

trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 torchrun train.py \
    -c configs/deimv2/ablation_experiments/deimv2_dinov3_s_offtype.yml \
    --use-amp --seed=0 2>&1 | tee "$LOGFILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
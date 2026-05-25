#!/bin/bash
set -o pipefail

mkdir -p nohup_log
LOGFILE=nohup_log/train_$(date +%Y%m%d_%H%M%S).log

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

# =========================
# 训练命令
# =========================

CUDA_VISIBLE_DEVICES=0 torchrun train.py \
    -c configs/deimv2/ablation_experiments/deimv2_dinov3_s_wheat_s78.yml \
    --use-amp --seed=0 2>&1 | tee "$LOGFILE"

# 关键：获取 torchrun 的真实退出码
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
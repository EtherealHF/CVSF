#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python stage1/alignment_train_v3.py \
  --config configs/stage1_alignment.yaml

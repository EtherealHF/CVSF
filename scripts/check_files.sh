#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

required=(
  "configs/inference_x2.yaml"
  "configs/stage1_alignment.yaml"
  "configs/stage2_cvsf.yaml"
  "checkpoints/stage1/image_40.pth"
  "checkpoints/stage1/eeg_40.pth"
  "checkpoints/stage2/model2.pkl"
  "assets/mm-realsr/de_net.pth"
  "pretrained/sd-turbo/model_index.json"
  "musiq_koniq_ckpt-e95806b9.pth"
  "niqe_modelparameters.mat"
)

for f in "${required[@]}"; do
  if [[ -e "$f" ]]; then
    echo "[OK] $f"
  else
    echo "[MISS] $f"
  fi
done

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 20659 src/train_cvsf.py \
  --base_config "configs/stage2_cvsf.yaml" \
  --sd_path="pretrained/sd-turbo" \
  --de_net_path="assets/mm-realsr/de_net.pth" \
  --stage1_img_encoder_ckpt="checkpoints/stage1/image_40.pth" \
  --stage1_backbone "resnet34" \
  --stage1_embed_dim 1024 \
  --stage1_input_size 224 \
  --prior_map_size 8 \
  --prior_channels 256 \
  --output_dir="train_output/cvsf_stage2" \
  --train_batch_size 1 \
  --mixed_precision bf16 \
  --eval_freq 500 \
  --save_val

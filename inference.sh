#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 20658 src/inference_cvsf.py \
  --base_config "configs/inference_x2.yaml" \
  --input_meta="examples/RELLISUR_testX2_example.txt" \
  --sd_path="pretrained/sd-turbo" \
  --de_net_path="assets/mm-realsr/de_net.pth" \
  --stage1_img_encoder_ckpt="checkpoints/stage1/image_40.pth" \
  --stage1_backbone "resnet34" \
  --stage1_embed_dim 1024 \
  --stage1_input_size 224 \
  --prior_map_size 8 \
  --prior_channels 256 \
  --output_dir="outputs/x2_test" \
  --pretrained_path="checkpoints/stage2/model2.pkl" \
  --infer_full_image \
  --tile_size 512 \
  --tile_overlap 64

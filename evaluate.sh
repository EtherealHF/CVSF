#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python scripts/evaluate_whole_folder.py \
  --evaluator "scripts/evaluate_img_new.py" \
  -i "outputs/x2_test/pred" \
  -r "outputs/x2_test/gt" \
  --save_vis_dir "outputs/x2_test/eval" \
  --record_file "outputs/x2_test/metrics_all.txt"

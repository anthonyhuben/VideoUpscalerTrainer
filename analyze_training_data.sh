#!/bin/bash
# Analyze training data — v3.0
# Optimized for MacBook Air M4 (16GB unified memory)

cd VideoUpscaler
source venv/bin/activate

python analyze_training_data.py \
  --lr_dir ./data/lr_frames \
  --hr_dir ./data/hr_frames \
  --sample_size 500 \
  --seed 42

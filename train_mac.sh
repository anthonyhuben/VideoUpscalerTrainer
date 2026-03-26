#!/bin/bash
# OPTIMIZED for MacBook Air M4 15" (16GB)
# v3.0 - Fixed gradient flow + focused loss selection

set -euo pipefail

cd VideoUpscaler
source venv/bin/activate

# ============================================================================
# M4 Environment Optimization
# ============================================================================
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4    # M4 Air has 4P+6E cores; use perf cores only
export MKL_NUM_THREADS=4

# Clear filesystem cache (helps on macOS with large datasets)
sudo purge 2>/dev/null || true

python train_upscaler_testing.py \
  --lr_dir ./data/lr_frames \
  --hr_dir ./data/hr_frames \
  --pretrained ./base_models/2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth
  --output_dir ./checkpoints \
  \
  `# Optimized batch settings for M4` \
  --batch_size 4 \
  --accumulation_steps 3 \
  --patch_size 128 \
  \
  `# Improved learning rate` \
  --lr 1e-4 \
  --epochs 5 \
  \
  `# Reduced validation overhead` \
  --val_every 1 \
  --save_every 1 \
  --val_save_every 1 \
  --val_split 0.05 \
  \
  `# Disable compilation (not beneficial on M4)` \
  --no_compile \
  \
  `# Conservative loss weights (tested and stable)` \
  --perceptual_weight 1.00 \
  --ssim_weight 0.11 \
  --lab_weight 0.05 \
  --brightness_weight 0.05 \
  --highlight_weight 0.77 \
  \
  `# Experimental losses` \
  --contrast_weight 0.0 \
  --color_weight 0.0 \
  --tv_weight 0.0 \
  --safe_margin 0.0 \
  --local_weight 0.0 \
  --exposure_weight 0.0 \
  --hist_weight 0.0 \
  --log_tone_weight 0.0 \
  --drange_weight 0.0 \
  --hlgrad_weight 0.0 \
  --highfreq_weight 0.0 \
  --edge_weight 0.0 \
  --mean_lum_weight 0.0 \
  --percentile_weight 0.0
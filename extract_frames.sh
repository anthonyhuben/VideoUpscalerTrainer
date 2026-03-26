#!/bin/bash
# Optimized for MacBook Air M4 (16GB unified memory)

cd VideoUpscaler
source venv/bin/activate

# Clear filesystem cache (helps on macOS with large datasets)
sudo purge 2>/dev/null || true

python extract_frames.py \
    --lr_videos ./videos/1080p \
    --hr_videos ./videos/4k \
    --lr_output ./data/lr_frames_new \
    --hr_output ./data/hr_frames_new \
    --num_sequences 18 \
    --skip_start 13459 \
    --skip_end 899
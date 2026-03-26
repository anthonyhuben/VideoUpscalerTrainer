# Quick Start Guide - Video Upscaler Fine-tuning

## Fastest Way to Get Started

### 1. Install Dependencies (5 minutes)
```bash
pip install torch torchvision opencv-python numpy Pillow tqdm imageio

# Or use the requirements file:
pip install -r requirements.txt
```

### 2. Prepare Your Videos (10 minutes)
Put your paired videos in folders:
```
videos/
├── 1080p/
│   └── clip1.mp4
└── 4k/
    └── clip1.mp4
```

### 3. Extract Training Frames (5-10 minutes)
```bash
python extract_frames.py \
    --lr_videos ./videos/1080p \
    --hr_videos ./videos/4k \
    --lr_output ./data/lr_frames \
    --hr_output ./data/hr_frames \
    --frame_interval 30
```

### 4. Train the Model (2-8 hours depending on GPU)
```bash
python train_upscaler.py \
    --lr_dir ./data/lr_frames \
    --hr_dir ./data/hr_frames \
    --pretrained ./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth \
    --output_dir ./checkpoints \
    --epochs 100
```

### 5. Upscale New Videos (real-time to minutes)
```bash
python inference.py \
    --input ./my_1080p_video.mp4 \
    --output ./my_4k_video.mp4 \
    --model ./checkpoints/best_model.pth \
    --gpu
```

## Minimum Working Example

If you have just ONE paired video:
1. Extract 300+ frames from it (`--max_frames_per_video 300`)
2. Train for 50 epochs
3. Test on a short clip

You should see improvement even with minimal data!

## Common First-Time Issues

**"CUDA out of memory"**
→ Add `--batch_size 2` or `--batch_size 1` to training command

**"No paired images found"**
→ Make sure 1080p and 4K videos have the exact same filename

**Training seems stuck**
→ It's not! First epoch is always slowest. Check GPU usage with `nvidia-smi`

## Expected Results

- **After 10 epochs:** Model starts adapting to your data
- **After 50 epochs:** Noticeable improvement on your specific content
- **After 100 epochs:** Well-adapted, should produce high-quality results

## Need More Detail?

See the full README.md for comprehensive documentation!

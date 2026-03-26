# Video Upscaler Fine-tuning Guide

Complete pipeline for fine-tuning a 2x upscaler model on 1080p→4K video footage.

## Overview

This training setup allows you to:
1. Extract frames from paired 1080p and 4K videos
2. Fine-tune a pretrained upscaler model on your specific video style
3. Upscale new 1080p videos to 4K using your trained model

## Prerequisites

- NVIDIA GPU with at least 6GB VRAM (for training)
- Python 3.8+
- CUDA toolkit (for GPU training)
- Paired video footage (same content in 1080p and 4K)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended), install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step-by-Step Workflow

### Step 1: Prepare Your Video Data

Organize your videos like this:
```
videos/
├── 1080p/
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── clip3.mp4
└── 4k/
    ├── clip1.mp4
    ├── clip2.mp4
    └── clip3.mp4
```

**Important:** The 1080p and 4K versions must have the **same filename** and should be the same content.

### Step 2: Extract Frames from Videos

```bash
python extract_frames.py \
    --lr_videos ./videos/1080p \
    --hr_videos ./videos/4k \
    --lr_output ./data/lr_frames \
    --hr_output ./data/hr_frames \
    --frame_interval 30 \
    --max_frames_per_video 300
```

**Parameters:**
- `--frame_interval 30`: Extracts 1 frame per second (at 30fps video)
- `--max_frames_per_video 300`: Limits to 300 frames per video (optional)

**Recommended dataset size:**
- Minimum: 500-1000 frame pairs
- Good: 2000-5000 frame pairs
- Excellent: 10,000+ frame pairs

### Step 3: Train the Model

```bash
python train_upscaler.py \
    --lr_dir ./data/lr_frames \
    --hr_dir ./data/hr_frames \
    --pretrained ./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --patch_size 128 \
    --lr 1e-4 \
    --epochs 100 \
    --save_every 10
```

**Training Parameters:**

- `--batch_size`: 
  - 4-8 for GPUs with 6-8GB VRAM
  - 2-4 for GPUs with 4-6GB VRAM
  - 1 for CPUs or lower VRAM (slow)

- `--patch_size`: Size of image patches for training
  - 128: Lower VRAM, faster training
  - 192: Balanced (recommended)
  - 256: Higher quality, needs more VRAM

- `--lr`: Learning rate
  - 1e-4: Standard for fine-tuning (recommended)
  - 2e-5: Very conservative (slower, more stable)
  - 5e-4: Aggressive (faster but less stable)

- `--epochs`: Number of training epochs
  - 50-100: Good starting point
  - Monitor validation loss to avoid overfitting

**Training Tips:**

1. **Monitor the validation loss** - training should stop when validation loss stops improving
2. **Start conservative** - use lower learning rate and increase if training is too slow
3. **Use the best_model.pth** - this is automatically saved when validation loss improves
4. **Expect 2-8 hours** training time depending on dataset size and GPU

### Step 4: Test Your Trained Model

**Upscale a video:**
```bash
python inference.py \
    --input ./test_video_1080p.mp4 \
    --output ./test_video_4k.mp4 \
    --model ./checkpoints/best_model.pth \
    --gpu
```

**Upscale a single image:**
```bash
python inference.py \
    --input ./test_image_1080p.png \
    --output ./test_image_4k.png \
    --model ./checkpoints/best_model.pth \
    --gpu
```

**Parameters:**
- `--tile_size 512`: Adjust based on GPU VRAM (lower = less memory)
- `--gpu`: Use GPU acceleration (remove for CPU inference)

## Understanding Your Results

### Training Metrics

- **Train Loss**: Should steadily decrease
- **Val Loss**: Should decrease and plateau (watch for overfitting if it increases)
- **Best model**: Saved automatically when validation loss improves

### Typical Loss Values

- Start: 0.01-0.05 (pretrained model)
- Good: 0.005-0.015 (well fine-tuned)
- Excellent: 0.003-0.008 (very well adapted)

### Avoiding Overfitting

Signs of overfitting:
- Training loss continues to decrease while validation loss increases
- Model works great on training videos but poorly on new videos

Solutions:
- Stop training earlier (use best_model.pth, not final_model.pth)
- Add more training data
- Use stronger data augmentation
- Reduce epochs

## Advanced Options

### Resume Training from Checkpoint

Modify the training script to load from a checkpoint:

```python
# In train_upscaler.py, add:
checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Fine-tune Only Certain Layers

To freeze early layers and only train later ones:

```python
# In train_upscaler.py, before creating optimizer:
# Freeze early layers
for name, param in model.named_parameters():
    if 'conv_first' in name or 'body.0' in name or 'body.1' in name:
        param.requires_grad = False

# Only optimize unfrozen parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
```

### Adjust Tile Size for Different VRAM

For inference on different GPUs:
- 24GB VRAM: `--tile_size 1024`
- 12GB VRAM: `--tile_size 512` (default)
- 8GB VRAM: `--tile_size 384`
- 6GB VRAM: `--tile_size 256`
- 4GB VRAM: `--tile_size 192`

## Troubleshooting

### Out of Memory (OOM) Errors

**During Training:**
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--patch_size` (try 96 or 64)
- Use gradient accumulation (modify script)

**During Inference:**
- Reduce `--tile_size`
- Use CPU instead of GPU (slower but works)

### Poor Results

1. **Model not improving:**
   - Check that frame pairs match correctly
   - Verify images are loading properly
   - Try increasing learning rate slightly

2. **Artifacts in output:**
   - Train longer (more epochs)
   - Use more training data
   - Check if pretrained model loads correctly

3. **Blurry output:**
   - Common with pixel loss only
   - Consider adding perceptual loss (advanced)
   - Ensure you're using best_model.pth, not early checkpoint

### Frame Extraction Issues

- **Videos not found:** Ensure filenames match exactly between 1080p and 4K folders
- **Too few frames:** Reduce `--frame_interval` or remove `--max_frames_per_video` limit
- **Wrong resolution:** Verify input videos are actually 1080p and 4K

## File Structure

```
project/
├── train_upscaler.py          # Main training script
├── extract_frames.py          # Video frame extraction
├── inference.py               # Upscale videos/images
├── requirements.txt           # Dependencies
├── 2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth  # Pretrained model
├── videos/
│   ├── 1080p/                # Your 1080p videos
│   └── 4k/                   # Your 4K videos
├── data/
│   ├── lr_frames/            # Extracted 1080p frames
│   └── hr_frames/            # Extracted 4K frames
└── checkpoints/              # Saved models
    ├── best_model.pth        # Best performing model
    ├── final_model.pth       # Last epoch model
    └── checkpoint_epoch_*.pth # Periodic checkpoints
```

## Performance Expectations

### Training Speed (per epoch)
- RTX 4090: ~2-5 minutes (1000 images, batch_size=8)
- RTX 3080: ~5-10 minutes (1000 images, batch_size=4)
- RTX 2060: ~10-20 minutes (1000 images, batch_size=2)
- CPU: ~2-4 hours (not recommended)

### Inference Speed (1080p→4K)
- RTX 4090: ~30-60 fps
- RTX 3080: ~15-30 fps
- RTX 2060: ~8-15 fps
- CPU: ~1-3 fps

## Next Steps

1. **Start small:** Test with 2-3 short video clips first
2. **Validate results:** Compare outputs to original 4K footage
3. **Iterate:** Adjust training parameters based on results
4. **Scale up:** Add more data once you're happy with the approach

## Questions?

Common questions:
- **How much data do I need?** At least 500-1000 frames, ideally 2000+
- **How long to train?** Usually 50-100 epochs, 2-8 hours total
- **Can I use this on different content?** You'll need to retrain on new data
- **What if I only have 720p→1080p?** The model works for any 2x upscaling

Good luck with your training! 🚀

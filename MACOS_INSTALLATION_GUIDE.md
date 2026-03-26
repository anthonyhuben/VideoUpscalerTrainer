# macOS Installation Guide - Video Upscaler Fine-tuning

Complete installation guide for macOS (including macOS Sequoia/Tahoe and Apple Silicon Macs)

## System Requirements

- macOS 10.15 (Catalina) or later
- 8GB RAM minimum (16GB+ recommended for training)
- 20GB free disk space
- Python 3.8 or later

**Note for Apple Silicon Macs (M1/M2/M3/M4):**
- PyTorch has native Apple Silicon support with Metal Performance Shaders (MPS)
- Training will use your GPU automatically
- Performance is good, though not as fast as NVIDIA GPUs

## Step-by-Step Installation

### Step 1: Install Homebrew (if not already installed)

Open **Terminal** (Applications → Utilities → Terminal) and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. If you already have Homebrew, skip this step.

### Step 2: Install Python 3

```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Verify installation
python3 --version
```

You should see something like `Python 3.11.x`

### Step 3: Create a Project Folder

```bash
# Create a folder for your project
mkdir ~/VideoUpscaler
cd ~/VideoUpscaler

# Download your files here (the 7 Python/text files I created)
# Also put your .pth model file here
```

### Step 4: Create a Virtual Environment (Recommended)

This keeps your project dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) at the start of your prompt
```

**Important:** You'll need to run `source venv/bin/activate` every time you open a new Terminal window to work on this project.

### Step 5: Install PyTorch for macOS

**For Apple Silicon Macs (M1/M2/M3/M4):**
```bash
pip install --upgrade pip
pip install torch torchvision
```

**For Intel Macs:**
```bash
pip install --upgrade pip
pip install torch torchvision
```

PyTorch will automatically use:
- **MPS (Metal Performance Shaders)** on Apple Silicon for GPU acceleration
- **CPU** on Intel Macs

### Step 6: Install Other Dependencies

```bash
pip install opencv-python numpy Pillow tqdm
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 7: Verify Installation

```bash
python3 << 'EOF'
import torch
import cv2
import numpy as np
from PIL import Image

print("✅ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Check for MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    print("🚀 Metal Performance Shaders (MPS) available - GPU acceleration enabled!")
elif torch.cuda.is_available():
    print("🚀 CUDA available - GPU acceleration enabled!")
else:
    print("⚠️  Using CPU only - training will be slower")

EOF
```

If you see "All packages installed successfully!" you're ready to go!

## macOS-Specific Modifications

### Using Apple Silicon GPU (M1/M2/M3/M4)

The training script will automatically detect and use your Apple Silicon GPU. However, you may need to make small modifications for optimal performance:

**Option 1: Automatic (Recommended)**
The scripts should work as-is. PyTorch will automatically use MPS when available.

**Option 2: Force MPS Usage**
If you want to explicitly use MPS, modify the device selection in the scripts:

```python
# In train_upscaler.py, replace the device line:
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Performance Expectations on Mac

**Apple Silicon (M1/M2/M3/M4):**
- M1 Mac: ~5-15 minutes per epoch (1000 images)
- M2/M3 Mac: ~3-10 minutes per epoch
- M4 Mac: ~2-8 minutes per epoch
- Inference: 5-20 fps for 1080p→4K

**Intel Mac:**
- Training: ~20-40 minutes per epoch (CPU only)
- Inference: 1-5 fps for 1080p→4K
- Consider using a smaller batch size (--batch_size 1)

### Recommended Settings for Mac

**For Training (Apple Silicon):**
```bash
python train_upscaler.py \
    --lr_dir ./data/lr_frames \
    --hr_dir ./data/hr_frames \
    --pretrained ./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth \
    --batch_size 4 \
    --patch_size 128 \
    --epochs 100
```

**For Training (Intel Mac - CPU only):**
```bash
python train_upscaler.py \
    --lr_dir ./data/lr_frames \
    --hr_dir ./data/hr_frames \
    --pretrained ./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth \
    --batch_size 1 \
    --patch_size 96 \
    --epochs 50
```

**For Inference:**
```bash
# The script will auto-detect GPU
python inference.py \
    --input ./video.mp4 \
    --output ./video_4k.mp4 \
    --model ./checkpoints/best_model.pth \
    --gpu
```

## File Organization on Mac

```
~/VideoUpscaler/                          # Your main project folder
├── venv/                                 # Virtual environment (created by you)
├── train_upscaler.py                     # Downloaded from Claude
├── extract_frames.py                     # Downloaded from Claude
├── inference.py                          # Downloaded from Claude
├── evaluate.py                           # Downloaded from Claude
├── requirements.txt                      # Downloaded from Claude
├── README.md                             # Downloaded from Claude
├── QUICKSTART.md                         # Downloaded from Claude
├── 2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth  # Your model
├── videos/
│   ├── 1080p/                           # Your 1080p videos
│   │   └── clip1.mp4
│   └── 4k/                              # Your 4K videos
│       └── clip1.mp4
├── data/                                # Created by extract_frames.py
│   ├── lr_frames/
│   └── hr_frames/
└── checkpoints/                         # Created by train_upscaler.py
    ├── best_model.pth
    └── final_model.pth
```

## Common macOS Issues and Solutions

### Issue 1: "python: command not found"
**Solution:** Use `python3` instead of `python`:
```bash
python3 train_upscaler.py --lr_dir ...
```

### Issue 2: "Permission denied"
**Solution:** Don't use `sudo`. Use virtual environment instead:
```bash
source venv/bin/activate
pip install ...
```

### Issue 3: "No module named 'cv2'"
**Solution:** Reinstall OpenCV:
```bash
pip uninstall opencv-python
pip install opencv-python-headless  # Use headless version on Mac
```

### Issue 4: "MPS backend out of memory"
**Solution:** Reduce batch size and patch size:
```bash
python train_upscaler.py --batch_size 2 --patch_size 96 ...
```

### Issue 5: "xcrun: error: invalid active developer path"
**Solution:** Install Xcode Command Line Tools:
```bash
xcode-select --install
```

### Issue 6: Video codec issues
**Solution:** Install ffmpeg:
```bash
brew install ffmpeg
```

## Step-by-Step First Run

1. **Open Terminal** (⌘+Space, type "Terminal")

2. **Navigate to your project:**
   ```bash
   cd ~/VideoUpscaler
   source venv/bin/activate
   ```

3. **Put your videos in folders:**
   ```bash
   mkdir -p videos/1080p videos/4k
   # Now drag your videos into these folders using Finder
   ```

4. **Extract frames:**
   ```bash
   python3 extract_frames.py \
       --lr_videos ./videos/1080p \
       --hr_videos ./videos/4k \
       --lr_output ./data/lr_frames \
       --hr_output ./data/hr_frames \
       --frame_interval 30
   ```

5. **Start training:**
   ```bash
   python3 train_upscaler.py \
       --lr_dir ./data/lr_frames \
       --hr_dir ./data/hr_frames \
       --pretrained ./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth \
       --output_dir ./checkpoints \
       --batch_size 4 \
       --epochs 100
   ```

6. **Monitor progress:**
   - Watch the terminal for loss values decreasing
   - Training will take 2-8 hours depending on your Mac
   - You can stop anytime with Ctrl+C

7. **Test your model:**
   ```bash
   python3 inference.py \
       --input ./videos/1080p/clip1.mp4 \
       --output ./output_4k.mp4 \
       --model ./checkpoints/best_model.pth \
       --gpu
   ```

## Optimizing for Your Mac

### For M1/M2/M3/M4 Macs:
- Use `--batch_size 4` or `--batch_size 8`
- Use `--patch_size 128` or `--patch_size 192`
- Training should be reasonably fast

### For Intel Macs:
- Use `--batch_size 1` or `--batch_size 2`
- Use `--patch_size 96`
- Consider training overnight
- Or use a cloud GPU service (Google Colab, AWS, etc.)

### Battery Life Tips:
- Plug in your Mac when training
- Close other applications
- Training will make your Mac run warm - this is normal
- Use Activity Monitor to watch resource usage

## Getting Help

If something isn't working:

1. **Check Python version:** `python3 --version` (should be 3.8+)
2. **Check if packages are installed:** `pip list`
3. **Make sure virtual environment is activated:** You should see `(venv)` in your prompt
4. **Check file locations:** `ls -la` to see what's in your folder

## Next Steps

Once everything is installed:
1. Read the QUICKSTART.md for a fast walkthrough
2. Read the README.md for detailed documentation
3. Start with a small test dataset (1-2 videos)
4. Scale up once you're comfortable with the workflow

Good luck with your training! 🚀

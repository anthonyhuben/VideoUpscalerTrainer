"""
Inference script to upscale 1080p video to 4K using trained model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import imageio

from train_upscaler import SuperUltraCompact


def load_model(checkpoint_path, device):
    model = SuperUltraCompact(nf=24, nc=8, scale=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    
    # ✅ Add strict mode with error reporting
    try:
        result = model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"❌ CRITICAL: Checkpoint mismatch!\n{e}")
        print("\nTrying non-strict loading...")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"⚠️  Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"⚠️  Unexpected keys: {result.unexpected_keys}")
    
    model = model.to(device)
    model.eval()
    return model

# ✅ ADD THE BLEND_TILES FUNCTION HERE
def blend_tiles(output, tile_output, out_y_start, out_y_end, 
                out_x_start, out_x_end, tile_pad, scale):
    """Blend overlapping tile regions to avoid seams"""
    blend_width = tile_pad * scale
    
    # Create blend mask (feathered edges)
    h = out_y_end - out_y_start
    w = out_x_end - out_x_start
    
    if blend_width > 0 and blend_width < w and blend_width < h:
        # Left blend
        if out_x_start > 0:
            for x in range(blend_width):
                alpha = x / blend_width
                output[out_y_start:out_y_end, out_x_start + x] = \
                    (1 - alpha) * output[out_y_start:out_y_end, out_x_start + x] + \
                    alpha * tile_output[:, x]
        
        # Top blend
        if out_y_start > 0:
            for y in range(blend_width):
                alpha = y / blend_width
                output[out_y_start + y, out_x_start:out_x_end] = \
                    (1 - alpha) * output[out_y_start + y, out_x_start:out_x_end] + \
                    alpha * tile_output[y, :]
    
    # Place tile
    output[out_y_start:out_y_end, out_x_start:out_x_end] = tile_output


def process_frame(model, frame, device, tile_size=512, tile_pad=10):
    """
    Process a single frame with tiling to handle large images
    
    Args:
        model: Upscaler model
        frame: Input frame (numpy array, BGR)
        device: torch device
        tile_size: Size of tiles for processing
        tile_pad: Padding for tiles to avoid edge artifacts
    """
    # Convert BGR to RGB and normalize to [0, 1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    h, w = img.shape[:2]
    
    # If image is small enough, process directly
    if h <= tile_size and w <= tile_size:
        # Convert to tensor
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Upscale
        with torch.no_grad():
            output = model(img_tensor)

        # ✅ Add validation
        if torch.isnan(output).any():
            print("❌ ERROR: Model output contains NaN values!")
            return None

        if torch.isinf(output).any():
            print("❌ ERROR: Model output contains Inf values!")
            return None

        # Check shape
        if output.shape[1] != 3:
            print(f"❌ ERROR: Expected 3 channels, got {output.shape[1]}")
            print(f"   Full output shape: {output.shape}")
            return None

        output = output.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))

        # ✅ Clamp BEFORE converting to uint8
        output = np.clip(output, 0, 1)  # Ensure [0, 1] range
        output = (output * 255.0).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    # Process with tiles
    scale = 2  # 2x upscaler
    output_h, output_w = h * scale, w * scale
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    tiles_x = (w + tile_size - 1) // tile_size
    tiles_y = (h + tile_size - 1) // tile_size
    
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Calculate tile coordinates with padding
            x_start = max(0, j * tile_size - tile_pad)
            y_start = max(0, i * tile_size - tile_pad)
            x_end = min(w, (j + 1) * tile_size + tile_pad)
            y_end = min(h, (i + 1) * tile_size + tile_pad)
            
            # Extract tile
            tile = img[y_start:y_end, x_start:x_end, :]
            
            # Convert to tensor
            tile_tensor = torch.from_numpy(np.transpose(tile, (2, 0, 1))).float()
            tile_tensor = tile_tensor.unsqueeze(0).to(device)
            
            # Upscale
            with torch.no_grad():
                tile_output = model(tile_tensor)
            
            # ✅ Validate output
            if torch.isnan(tile_output).any() or torch.isinf(tile_output).any():
                print(f"⚠️  Warning: Invalid output at tile ({i}, {j}), skipping")
                continue
            
            # Convert back to numpy
            tile_output = tile_output.squeeze(0).cpu().numpy()
            tile_output = np.transpose(tile_output, (1, 2, 0))
            
            # ✅ Ensure valid range
            tile_output = np.clip(tile_output, 0, 1)
            tile_output = (tile_output * 255.0).astype(np.uint8)
            
            # Calculate output position (accounting for padding)
            out_x_start = j * tile_size * scale
            out_y_start = i * tile_size * scale
            out_x_end = min(output_w, (j + 1) * tile_size * scale)
            out_y_end = min(output_h, (i + 1) * tile_size * scale)
            
            # Calculate crop for tile (to remove padding)
            crop_x_start = (out_x_start - x_start * scale)
            crop_y_start = (out_y_start - y_start * scale)
            crop_x_end = crop_x_start + (out_x_end - out_x_start)
            crop_y_end = crop_y_start + (out_y_end - out_y_start)
            
            # ✅ Place tile with blending to avoid seams
            blend_tiles(output,
                        tile_output[crop_y_start:crop_y_end, crop_x_start:crop_x_end],
                        out_y_start, out_y_end, out_x_start, out_x_end, 
                        tile_pad, scale)
    
    # Convert RGB to BGR
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output


def upscale_video(model, input_video, output_video, device, tile_size=512):
    """Upscale an entire video"""
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {input_video}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Output resolution (2x upscale)
    out_width = width * 2
    out_height = height * 2
    
    print(f"\nOutput video: {output_video}")
    print(f"  Resolution: {out_width}x{out_height}")
    
    # ✅ Use imageio for better color space handling
    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p'
    )
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Upscale frame
        upscaled_frame = process_frame(model, frame, device, tile_size)
        
        # ✅ Handle failed frames
        if upscaled_frame is None:
            print(f"Frame processing failed at frame {frame_num}")
            continue
        
        # Write to output
        writer.append_data(upscaled_frame)
        
        frame_num += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    writer.close()
    
    print(f"\nUpscaling complete! Saved to {output_video}")


def upscale_image(model, input_image, output_image, device):
    """Upscale a single image"""
    # Read image
    img = cv2.imread(str(input_image))
    
    if img is None:
        print(f"Error: Could not read image {input_image}")
        return
    
    print(f"Input image: {input_image}")
    print(f"  Resolution: {img.shape[1]}x{img.shape[0]}")
    
    # Upscale
    upscaled = process_frame(model, img, device)
    
    # ✅ Handle failed processing
    if upscaled is None:
        print("ERROR: Image processing failed!")
        return
    
    # Save
    cv2.imwrite(str(output_image), upscaled)
    
    print(f"Output image: {output_image}")
    print(f"  Resolution: {upscaled.shape[1]}x{upscaled.shape[0]}")
    print("Upscaling complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Upscale video/image using trained model'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input video or image file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video or image file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for processing (lower = less VRAM)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # ✅ Device detection with MPS support
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using device: MPS (Apple Silicon GPU)")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (slower)")
    
    # Load model
    model = load_model(args.model, device)
    
    # Check if input is video or image
    input_path = Path(args.input)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    
    if input_path.suffix.lower() in video_extensions:
        # Process video
        upscale_video(model, args.input, args.output, device, args.tile_size)
    else:
        # Process image
        upscale_image(model, args.input, args.output, device)


if __name__ == '__main__':
    main()
    
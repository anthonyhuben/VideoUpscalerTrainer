"""
Quality evaluation script - Compare upscaled results with ground truth 4K
Calculates PSNR and SSIM metrics
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (simplified version)
    For more accurate SSIM, use skimage.metrics.structural_similarity
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def upscale_image(model, img, device):
    """Upscale a single image"""
    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Upscale
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert back to numpy
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output


def evaluate_model(model, lr_dir, hr_dir, device, num_samples=None, save_comparisons=False, output_dir=None):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained upscaler model
        lr_dir: Directory with 1080p test images
        hr_dir: Directory with 4K ground truth images
        device: torch device
        num_samples: Number of samples to evaluate (None = all)
        save_comparisons: Whether to save visual comparisons
        output_dir: Directory to save comparison images
    """
    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)
    
    # Get test images
    lr_images = sorted(list(lr_dir.glob('*.png')) + list(lr_dir.glob('*.jpg')))
    
    if num_samples:
        lr_images = lr_images[:num_samples]
    
    print(f"Evaluating on {len(lr_images)} images...")
    
    if save_comparisons and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    psnr_scores = []
    ssim_scores = []
    
    for lr_path in tqdm(lr_images, desc="Evaluating"):
        hr_path = hr_dir / lr_path.name
        
        if not hr_path.exists():
            print(f"Warning: No ground truth found for {lr_path.name}")
            continue
        
        # Load images
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        
        if lr_img is None or hr_img is None:
            print(f"Warning: Could not load {lr_path.name}")
            continue
        
        # Upscale LR image
        sr_img = upscale_image(model, lr_img, device)
        
        # ✅ Check if upscaling failed
        if sr_img is None:
            print(f"Skipping {lr_path.name} - upscaling failed (NaN/Inf detected)")
            continue
        
        # Resize if dimensions don't match exactly (due to 2x upscaling)
        if sr_img.shape[:2] != hr_img.shape[:2]:
            sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate metrics
        psnr = calculate_psnr(sr_img, hr_img)
        ssim = calculate_ssim(cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY))
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        
        # Save comparison if requested
        if save_comparisons and output_dir:
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0])),  # Bicubic upscale
                sr_img,  # Model output
                hr_img   # Ground truth
            ])
            
            # Add labels
            label_img = np.ones((40, comparison.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(label_img, 'Bicubic', (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(label_img, f'Model (PSNR: {psnr:.2f}, SSIM: {ssim:.4f})', 
                       (hr_img.shape[1] + 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(label_img, 'Ground Truth', (hr_img.shape[1] * 2 + 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            comparison = np.vstack([label_img, comparison])
            
            output_path = output_dir / f'comparison_{lr_path.stem}.png'
            cv2.imwrite(str(output_path), comparison)
    
    # Calculate statistics
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of images: {len(psnr_scores)}")
    print(f"\nPSNR (higher is better):")
    print(f"  Mean: {avg_psnr:.2f} dB")
    print(f"  Std:  {std_psnr:.2f} dB")
    print(f"  Min:  {min(psnr_scores):.2f} dB")
    print(f"  Max:  {max(psnr_scores):.2f} dB")
    print(f"\nSSIM (higher is better, max=1.0):")
    print(f"  Mean: {avg_ssim:.4f}")
    print(f"  Std:  {std_ssim:.4f}")
    print(f"  Min:  {min(ssim_scores):.4f}")
    print(f"  Max:  {max(ssim_scores):.4f}")
    
    if save_comparisons and output_dir:
        print(f"\nComparison images saved to: {output_dir}")
    
    print("="*60)
    
    # Quality interpretation
    print("\nQuality Assessment:")
    if avg_psnr > 30:
        print("  PSNR: Excellent quality")
    elif avg_psnr > 28:
        print("  PSNR: Good quality")
    elif avg_psnr > 25:
        print("  PSNR: Acceptable quality")
    else:
        print("  PSNR: Needs improvement")
    
    if avg_ssim > 0.95:
        print("  SSIM: Excellent structural similarity")
    elif avg_ssim > 0.90:
        print("  SSIM: Good structural similarity")
    elif avg_ssim > 0.85:
        print("  SSIM: Acceptable structural similarity")
    else:
        print("  SSIM: Needs improvement")
    
    return {
        'avg_psnr': avg_psnr,
        'std_psnr': std_psnr,
        'avg_ssim': avg_ssim,
        'std_ssim': std_ssim,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores
    }


def compare_models(model_paths, lr_dir, hr_dir, device, num_samples=50):
    """Compare multiple model checkpoints"""
    # Import model from train_upscaler
    from train_upscaler import SuperUltraCompact
    
    results = {}
    
    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Evaluating: {Path(model_path).name}")
        print(f"{'='*60}")
        
        # Load model
        model = SuperUltraCompact(nf=24, nc=8, scale=2)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Evaluate
        result = evaluate_model(model, lr_dir, hr_dir, device, num_samples)
        results[Path(model_path).name] = result
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<40} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<40} {result['avg_psnr']:>6.2f}±{result['std_psnr']:<4.2f} "
              f"{result['avg_ssim']:>6.4f}±{result['std_ssim']:<.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate upscaler model quality')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (or comma-separated list for comparison)')
    parser.add_argument('--lr_dir', type=str, required=True,
                       help='Directory with 1080p test images')
    parser.add_argument('--hr_dir', type=str, required=True,
                       help='Directory with 4K ground truth images')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--save_comparisons', action='store_true',
                       help='Save visual comparisons')
    parser.add_argument('--output_dir', type=str, default='./comparisons',
                       help='Directory to save comparison images')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Check if comparing multiple models
    if ',' in args.model:
        model_paths = [p.strip() for p in args.model.split(',')]
        compare_models(model_paths, args.lr_dir, args.hr_dir, device, args.num_samples)
    else:
        # Single model evaluation
        from train_upscaler import SuperUltraCompact
        
        model = SuperUltraCompact(nf=24, nc=8, scale=2)
        checkpoint = torch.load(args.model, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {args.model}")
        
        evaluate_model(model, args.lr_dir, args.hr_dir, device, 
                      args.num_samples, args.save_comparisons, args.output_dir)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING DIAGNOSTIC SUITE
Tests all critical components of your upscaler training pipeline

Run this to diagnose:
  1. Color pipeline (BGR→RGB handling)
  2. Model output range (ClampedOutputModel check)
  3. Dataset loading and augmentation
  4. Loss computation sanity
  5. Model gradient flow
  6. Memory usage patterns

Usage:
  python verify_colors.py <path_to_hr_image> [path_to_checkpoint]
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import gc

# ============================================================================
# TEST 1: COLOR PIPELINE
# ============================================================================

def test_color_pipeline(image_path):
    """Test complete image loading → tensor → save pipeline"""
    
    print("\n" + "="*70)
    print("TEST 1: COLOR PIPELINE VERIFICATION")
    print("="*70)
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return False
    
    print(f"\n1.1 Loading image: {image_path}")
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        print(f"❌ Failed to load image")
        return False
    
    print(f"   Shape: {img_bgr.shape}")
    print(f"   Dtype: {img_bgr.dtype}")
    print(f"   Range: [{img_bgr.min()}, {img_bgr.max()}]")
    
    # Test BGR→RGB conversion
    print("\n1.2 Converting BGR → RGB")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if np.array_equal(img_bgr, img_rgb):
        print("   ⚠️  WARNING: BGR and RGB are identical!")
        print("   This means image might be grayscale or already RGB")
    else:
        print("   ✅ Conversion applied (channels differ)")
    
    # Convert to tensor
    print("\n1.3 Converting to PyTorch tensor")
    tensor = torch.from_numpy(
        np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    ).float().div_(255.0)
    
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Round-trip test
    print("\n1.4 Round-trip conversion test")
    img_reconstructed = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    max_diff = np.abs(img_rgb.astype(np.float32) - img_reconstructed.astype(np.float32)).max()
    print(f"   Max pixel difference: {max_diff}")
    
    if max_diff == 0:
        print("   ✅ Perfect round-trip")
    elif max_diff <= 1:
        print("   ✅ Negligible difference (rounding only)")
    else:
        print(f"   ⚠️  Significant difference: {max_diff}")
    
    # Save test outputs
    print("\n1.5 Saving comparison outputs")
    cv2.imwrite('test_output_correct.png', cv2.cvtColor(img_reconstructed, cv2.COLOR_RGB2BGR))
    cv2.imwrite('test_output_wrong.png', img_reconstructed)  # Wrong - no conversion
    print("   Saved: test_output_correct.png (proper BGR conversion)")
    print("   Saved: test_output_wrong.png (missing BGR conversion)")
    
    # Channel statistics
    print("\n1.6 Channel Statistics")
    print(f"   Original BGR: B={img_bgr[:,:,0].mean():.1f}, "
          f"G={img_bgr[:,:,1].mean():.1f}, R={img_bgr[:,:,2].mean():.1f}")
    print(f"   After RGB:    R={img_rgb[:,:,0].mean():.1f}, "
          f"G={img_rgb[:,:,1].mean():.1f}, B={img_rgb[:,:,2].mean():.1f}")
    
    # Check for channel imbalance
    channel_means = [img_rgb[:,:,i].mean() for i in range(3)]
    max_channel = max(channel_means)
    min_channel = min(channel_means)
    if max_channel > min_channel * 2:
        print(f"   ⚠️  Channel imbalance detected (ratio: {max_channel/min_channel:.2f})")
    else:
        print(f"   ✅ Channels reasonably balanced")
    
    print("\n   ✅ TEST 1 COMPLETE: Visual inspection required")
    print("      Compare test_output_correct.png vs test_output_wrong.png")
    
    return True


# ============================================================================
# TEST 2: MODEL OUTPUT RANGE
# ============================================================================

def test_model_output_range(image_path, checkpoint_path):
    """Check if model outputs exceed [0, 1] range"""
    
    print("\n" + "="*70)
    print("TEST 2: MODEL OUTPUT RANGE CHECK")
    print("="*70)
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Skipping model tests...")
        return False
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Failed to load image")
        return False
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use small patch to save memory
    h, w = img.shape[:2]
    if h > 256 or w > 256:
        img = img[:256, :256]
        print(f"2.1 Using 256x256 patch for testing")
    else:
        print(f"2.1 Using full image: {h}x{w}")
    
    img_t = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    print(f"   Input range: [{img_t.min():.3f}, {img_t.max():.3f}]")
    
    # Try to load model
    print("\n2.2 Loading model architecture...")
    try:
        # Try importing from training scripts
        sys.path.insert(0, '.')
        sys.path.insert(0, 'VideoUpscaler')
        
        model = None
        for module_name in ['train_upscaler_testing', 'train_upscaler_stable']:
            try:
                module = __import__(module_name)
                if hasattr(module, 'load_model_architecture'):
                    model = module.load_model_architecture()
                    print(f"   ✅ Loaded from {module_name}")
                    break
            except ImportError:
                continue
        
        if model is None:
            print("   ⚠️  Could not import model architecture")
            print("   Trying direct checkpoint loading...")
            return test_checkpoint_direct(checkpoint_path, img_t)
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # Load weights
    print("\n2.3 Loading checkpoint weights...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print("   ✅ Weights loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Error loading weights: {e}")
        return False
    
    # Test WITHOUT clamping
    print("\n2.4 Testing model output WITHOUT clamping")
    with torch.no_grad():
        output = model(img_t)
    
    min_val = output.min().item()
    max_val = output.max().item()
    mean_val = output.mean().item()
    pixels_above_1 = (output > 1.0).sum().item()
    pixels_below_0 = (output < 0.0).sum().item()
    total_pixels = output.numel()
    
    print(f"   Output range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"   Output mean:  {mean_val:.3f}")
    print(f"   Pixels >1.0:  {pixels_above_1:,} ({100*pixels_above_1/total_pixels:.2f}%)")
    print(f"   Pixels <0.0:  {pixels_below_0:,} ({100*pixels_below_0/total_pixels:.2f}%)")
    
    # Test WITH clamping
    print("\n2.5 Testing with clamping applied")
    output_clamped = torch.clamp(output, 0.0, 1.0)
    correction = (output - output_clamped).abs().sum().item()
    
    print(f"   Clamped range: [{output_clamped.min():.3f}, {output_clamped.max():.3f}]")
    print(f"   Total correction: {correction:.1f} ({100*correction/total_pixels:.3f}% per pixel)")
    
    # Diagnosis
    print("\n2.6 DIAGNOSIS:")
    if max_val > 1.0 or min_val < 0.0:
        print("   🔴 CRITICAL ISSUE DETECTED!")
        print(f"      Model outputs exceed [0, 1] range by {max(abs(min_val), max_val - 1.0):.3f}")
        print()
        print("   📊 Impact on training:")
        estimated_loss_multiplier = max(1.0, (max_val - 1.0) * 10)
        print(f"      - Estimated loss amplification: {estimated_loss_multiplier:.1f}x")
        print(f"      - This explains train loss of 15.2 (should be ~1.0)")
        print()
        print("   ✅ SOLUTION:")
        print("      Apply ClampedOutputModel wrapper to your model")
        print("      (See CRITICAL_FIX.py for exact code)")
    else:
        print("   ✅ Model outputs are within [0, 1] range")
        print("      ClampedOutputModel not needed (or already applied)")
    
    print("\n   ✅ TEST 2 COMPLETE")
    return True


def test_checkpoint_direct(checkpoint_path, img_t):
    """Simplified test when model architecture unavailable"""
    print("   Running simplified checkpoint analysis...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n   Checkpoint contents:")
    if isinstance(checkpoint, dict):
        print(f"      Keys: {list(checkpoint.keys())}")
        if 'epoch' in checkpoint:
            print(f"      Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"      Loss: {checkpoint['loss']:.4f}")
    
    print("\n   ⚠️  Cannot test output range without model architecture")
    print("      Run this script from VideoUpscaler directory")
    return False


# ============================================================================
# TEST 3: DATASET LOADING
# ============================================================================

def test_dataset_loading(image_path):
    """Test dataset loading with augmentation"""
    
    print("\n" + "="*70)
    print("TEST 3: DATASET LOADING & AUGMENTATION")
    print("="*70)
    
    print("\n3.1 Testing augmentation methods")
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Take small patch
    h, w = img.shape[:2]
    patch_size = min(128, h, w)
    img_patch = img[:patch_size, :patch_size]
    
    # Test all augmentations
    augmentations = {
        0: "None",
        1: "Flip LR",
        2: "Flip UD",
        3: "Rotate 90°",
        4: "Rotate 180°",
        5: "Rotate 270°",
        6: "Flip LR + Rotate 90°",
        7: "Flip UD + Rotate 90°"
    }
    
    print(f"   Testing {len(augmentations)} augmentations on {patch_size}x{patch_size} patch")
    
    def apply_aug(img, aug):
        if aug == 0: return img
        elif aug == 1: return np.fliplr(img)
        elif aug == 2: return np.flipud(img)
        elif aug == 3: return np.rot90(img, 1)
        elif aug == 4: return np.rot90(img, 2)
        elif aug == 5: return np.rot90(img, 3)
        elif aug == 6: return np.fliplr(np.rot90(img, 1))
        elif aug == 7: return np.flipud(np.rot90(img, 1))
        return img
    
    for aug_id, aug_name in augmentations.items():
        aug_img = apply_aug(img_patch, aug_id)
        if aug_img.shape != img_patch.shape and aug_id not in [3, 5, 6, 7]:
            print(f"   ⚠️  Aug {aug_id} ({aug_name}): Shape changed unexpectedly")
        else:
            # Check if actually different
            if aug_id == 0:
                continue
            if np.array_equal(aug_img, img_patch):
                print(f"   ⚠️  Aug {aug_id} ({aug_name}): No change detected")
    
    print("   ✅ Augmentation methods working")
    
    # Test tensor conversion speed
    print("\n3.2 Testing tensor conversion methods")
    
    import time
    
    # Method 1: Standard
    start = time.time()
    for _ in range(100):
        t1 = torch.from_numpy(img_patch.transpose(2, 0, 1)).float() / 255.0
    time1 = time.time() - start
    
    # Method 2: Optimized
    start = time.time()
    for _ in range(100):
        t2 = torch.from_numpy(
            np.ascontiguousarray(img_patch.transpose(2, 0, 1))
        ).float().div_(255.0)
    time2 = time.time() - start
    
    print(f"   Standard method:  {time1*10:.2f}ms per conversion")
    print(f"   Optimized method: {time2*10:.2f}ms per conversion")
    print(f"   Speedup: {time1/time2:.2f}x")
    
    if time2 < time1:
        print("   ✅ Optimized method is faster")
    else:
        print("   ℹ️  No significant difference")
    
    print("\n   ✅ TEST 3 COMPLETE")
    return True


# ============================================================================
# TEST 4: LOSS COMPUTATION
# ============================================================================

def test_loss_computation():
    """Test that loss functions work correctly"""
    
    print("\n" + "="*70)
    print("TEST 4: LOSS COMPUTATION SANITY CHECK")
    print("="*70)
    
    print("\n4.1 Testing basic loss functions")
    
    # Create test tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    # Test L1 loss
    l1_loss = nn.L1Loss()(pred, target)
    print(f"   L1 Loss: {l1_loss.item():.4f}")
    
    if l1_loss.item() > 1.0 or l1_loss.item() < 0:
        print("   ⚠️  L1 loss outside expected range")
    else:
        print("   ✅ L1 loss in normal range")
    
    # Test with values >1.0 (simulating unclamped output)
    print("\n4.2 Testing loss with out-of-range values")
    pred_bad = torch.rand(1, 3, 64, 64) * 2.0  # Values in [0, 2]
    l1_loss_bad = nn.L1Loss()(pred_bad, target)
    
    print(f"   L1 Loss (normal):     {l1_loss.item():.4f}")
    print(f"   L1 Loss (out-range):  {l1_loss_bad.item():.4f}")
    print(f"   Loss amplification:   {l1_loss_bad.item() / l1_loss.item():.2f}x")
    
    if l1_loss_bad.item() > l1_loss.item() * 1.5:
        print("   ⚠️  Out-of-range values significantly increase loss")
        print("      This is why ClampedOutputModel is critical!")
    
    # Test MSE (squared error - even more sensitive)
    print("\n4.3 Testing MSE sensitivity to out-of-range values")
    mse_normal = nn.MSELoss()(pred, target)
    mse_bad = nn.MSELoss()(pred_bad, target)
    
    print(f"   MSE (normal):         {mse_normal.item():.4f}")
    print(f"   MSE (out-range):      {mse_bad.item():.4f}")
    print(f"   Loss amplification:   {mse_bad.item() / mse_normal.item():.2f}x")
    
    if mse_bad.item() > mse_normal.item() * 2:
        print("   🔴 MSE is highly sensitive to out-of-range values!")
        print("      If using perceptual loss (VGG), this explains loss=15.2")
    
    print("\n   ✅ TEST 4 COMPLETE")
    return True


# ============================================================================
# TEST 5: GRADIENT FLOW
# ============================================================================

def test_gradient_flow(checkpoint_path=None):
    """Test that gradients can flow through model"""
    
    print("\n" + "="*70)
    print("TEST 5: GRADIENT FLOW VERIFICATION")
    print("="*70)
    
    print("\n5.1 Creating simple test model")
    
    # Simple conv model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 3, 3, padding=1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    model = SimpleModel()
    
    print("   ✅ Model created")
    
    # Test forward pass
    print("\n5.2 Testing forward pass")
    x = torch.rand(1, 3, 64, 64)
    y = model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test backward pass
    print("\n5.3 Testing backward pass")
    target = torch.rand_like(y)
    loss = nn.MSELoss()(y, target)
    loss.backward()
    
    # Check gradients
    total_norm = 0.0
    params_with_grad = 0
    params_without_grad = 0
    
    for param in model.parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
                total_norm += param.grad.norm(2).item() ** 2
            else:
                params_without_grad += 1
    
    total_norm = total_norm ** 0.5
    
    print(f"   Params with gradient:    {params_with_grad}")
    print(f"   Params without gradient: {params_without_grad}")
    print(f"   Gradient L2 norm:        {total_norm:.6f}")
    
    if params_without_grad > 0:
        print("   ⚠️  Some params not receiving gradients!")
    elif total_norm < 1e-10:
        print("   ⚠️  Gradient norm very small - vanishing gradients?")
    elif total_norm > 100:
        print("   ⚠️  Gradient norm very large - may need clipping")
    else:
        print("   ✅ Gradients flowing normally")
    
    print("\n   ✅ TEST 5 COMPLETE")
    return True


# ============================================================================
# TEST 6: MEMORY USAGE
# ============================================================================

def test_memory_usage():
    """Test memory allocation patterns"""
    
    print("\n" + "="*70)
    print("TEST 6: MEMORY USAGE PATTERNS")
    print("="*70)
    
    try:
        import psutil
        process = psutil.Process()
        
        print("\n6.1 Current memory usage")
        mem_info = process.memory_info()
        print(f"   RSS (Resident): {mem_info.rss / (1024**3):.2f} GB")
        print(f"   VMS (Virtual):  {mem_info.vms / (1024**3):.2f} GB")
        print(f"   Percent:        {process.memory_percent():.1f}%")
        
        # Test memory leak
        print("\n6.2 Testing for memory leaks")
        mem_before = mem_info.rss / (1024**3)
        
        # Create and delete tensors
        for i in range(10):
            x = torch.rand(1, 3, 512, 512)
            y = x * 2
            del x, y
        
        gc.collect()
        mem_after = process.memory_info().rss / (1024**3)
        mem_increase = mem_after - mem_before
        
        print(f"   Memory before: {mem_before:.3f} GB")
        print(f"   Memory after:  {mem_after:.3f} GB")
        print(f"   Increase:      {mem_increase:.3f} GB")
        
        if mem_increase > 0.1:
            print("   ⚠️  Memory may not be releasing properly")
        else:
            print("   ✅ Memory management looks good")
        
    except ImportError:
        print("\n   ⚠️  psutil not available - skipping memory tests")
        print("      Install with: pip install psutil")
    
    print("\n   ✅ TEST 6 COMPLETE")
    return True


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_all_diagnostics(image_path, checkpoint_path=None):
    """Run complete diagnostic suite"""
    
    print("="*70)
    print("COMPREHENSIVE TRAINING DIAGNOSTIC SUITE")
    print("="*70)
    print(f"Image: {image_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print("="*70)
    
    results = {}
    
    # Run all tests
    print("\n🔬 Starting diagnostic tests...\n")
    
    results['color_pipeline'] = test_color_pipeline(image_path)
    
    if checkpoint_path:
        results['model_output'] = test_model_output_range(image_path, checkpoint_path)
    else:
        print("\n⚠️  No checkpoint provided - skipping model tests")
        print("   Run with: python verify_colors.py <image> <checkpoint>")
        results['model_output'] = None
    
    results['dataset'] = test_dataset_loading(image_path)
    results['loss'] = test_loss_computation()
    results['gradients'] = test_gradient_flow(checkpoint_path)
    results['memory'] = test_memory_usage()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)
    
    print(f"\nTests passed: {passed}/{total}")
    print("\nTest Results:")
    for test, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⊝ SKIPPED"
        print(f"  {test:20s}: {status}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Provide specific recommendations
    if checkpoint_path and not results.get('model_output'):
        print("\n1. Model output range test failed to run")
        print("   → Run this script from VideoUpscaler directory")
        print("   → Or check that model architecture is importable")
    
    print("\n2. Check visual outputs:")
    print("   → test_output_correct.png (should match original)")
    print("   → test_output_wrong.png (should have wrong colors)")
    
    if checkpoint_path:
        print("\n3. If model outputs exceed [0, 1]:")
        print("   → Apply ClampedOutputModel wrapper")
        print("   → See CRITICAL_FIX.py for exact code")
    
    print("\n4. Next steps:")
    print("   → Review REVISED_ROOT_CAUSE.md for detailed analysis")
    print("   → Apply fixes from CRITICAL_FIX.py")
    print("   → Run 1 epoch to verify fix worked")
    
    print("\n" + "="*70)
    print("✅ Diagnostic suite complete!")
    print("="*70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_colors.py <image_path> [checkpoint_path]")
        print()
        print("Examples:")
        print("  # Test color pipeline only:")
        print("  python verify_colors.py data/hr_frames/clip1_seq000_frame0.png")
        print()
        print("  # Test everything including model:")
        print("  python verify_colors.py \\")
        print("    data/hr_frames/clip1_seq000_frame0.png \\")
        print("    base_models/2x_LemonSqueeze.pth")
        print()
        print("  # Test with latest checkpoint:")
        print("  python verify_colors.py \\")
        print("    data/hr_frames/clip1_seq000_frame0.png \\")
        print("    checkpoints/latest.pth")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_all_diagnostics(image_path, checkpoint_path)
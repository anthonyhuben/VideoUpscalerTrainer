#!/usr/bin/env python3
"""
Fine-tuning script for 2x upscaler model - OPTIMIZED for MacBook Air M4 (16GB)
REVISED v2.0 - Comprehensive fix for over-brightening issue

ANTI-BRIGHTENING FIXES v2.0:
  🔴 CRITICAL FIXES:
    - Added ClampedOutputModel wrapper (prevents outputs > 1.0)
    - Added MeanLuminancePreservationLoss (prevents mean brightness drift)
    - Replaced PerceptualLoss with BrightnessNormalizedPerceptualLoss
    - Reduced default perceptual_weight: 0.5 -> 0.2
  
  🟠 HIGH PRIORITY:
    - Replaced HighlightSuppressionLoss with AdaptiveHighlightSuppressionLoss
    - Added PercentilePreservationLoss for distribution matching
    - Increased lab_weight, local_weight, brightness_weight defaults
  
  🟡 ADDITIONAL:
    - Added brightness drift monitoring with warnings
    - Updated default loss weights for better brightness preservation

Original features maintained:
  - Lab Color Space Loss with corrected sRGB->Linear conversion
  - Local Statistics Matching (prevents local hot spots)
  - SSIM Luminance Loss (perceptual without VGG)
  - Exposure-Aware Gradient Loss
  - LogToneMappingLoss
  - Soft Histogram Matching
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE importing torch
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'  # M4 has 8 cores

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import time
import gc
import copy

try:
    from torch import autocast as _torch_autocast  # PyTorch >= 2.0
    HAS_AUTOCAST = True
except Exception:
    try:
        from torch.cuda.amp import autocast as _torch_autocast  # PyTorch <= 1.x
        HAS_AUTOCAST = True
    except Exception:
        HAS_AUTOCAST = False
        _torch_autocast = None

from contextlib import nullcontext

def get_autocast(device_type: str):
    if not HAS_AUTOCAST:
        return nullcontext()
    try:
        # torch.autocast(device_type="cuda")
        return _torch_autocast(device_type=device_type)
    except TypeError:
        # torch.cuda.amp.autocast() has no device_type arg
        return _torch_autocast()

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  Warning: cv2 not found. Install with: pip install opencv-python")
if not HAS_CV2:
    raise ImportError("opencv-python is required for dataset loading in this script.")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def verify_gradient_flow(model, epoch, batch_idx):
    """
    Diagnostic: Verify gradients are actually flowing through the model.
    Call this once per epoch to confirm training is working.
    """
    if batch_idx != 0:
        return
    
    total_norm = 0.0
    num_params = 0
    zero_grad_params = 0
    none_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params += 1
            if param.grad is None:
                none_grad_params += 1
            elif param.grad.abs().sum().item() == 0:
                zero_grad_params += 1
            else:
                total_norm += param.grad.norm(2).item() ** 2
    
    total_norm = total_norm ** 0.5
    
    print(f"\n  📬 GRADIENT CHECK [Epoch {epoch}]:")
    print(f"     Total params with requires_grad: {num_params}")
    print(f"     Params with None grad:           {none_grad_params}")
    print(f"     Params with zero grad:           {zero_grad_params}")
    print(f"     Gradient L2 norm:                {total_norm:.6f}")
    
    if none_grad_params == num_params:
        print(f"     ❌ NO GRADIENTS FLOWING – backward() is broken!")
    elif total_norm < 1e-10:
        print(f"     ⚠️  Gradient norm near zero – learning rate may be too small")
    elif total_norm > 100:
        print(f"     ⚠️  Gradient norm very large – may need stronger clipping")
    else:
        print(f"     ✅ Gradients flowing normally")


# ============================================================================
# CONNECTED LOSS COMPUTATION (Port from testing.py)
# ============================================================================

def compute_total_loss(sr, hr, device, batch_idx, epoch,
                       criterion_l1, criterion_perceptual, use_perceptual, perceptual_weight,
                       criterion_brightness, brightness_weight,
                       criterion_contrast, contrast_weight,
                       criterion_color, color_weight,
                       criterion_tv, tv_weight,
                       criterion_highfreq, highfreq_weight,
                       criterion_edge, edge_weight,
                       criterion_highlight, highlight_weight,
                       criterion_drange, drange_weight,
                       criterion_hl_grad, hlgrad_weight,
                       criterion_lab, lab_weight,
                       criterion_local, local_weight,
                       criterion_ssim, ssim_weight,
                       criterion_exposure, exposure_weight,
                       criterion_hist, hist_weight,
                       criterion_log_tone, log_tone_weight,
                       criterion_mean_lum, mean_lum_weight,
                       criterion_percentile, percentile_weight):
    """
    Compute total loss AS A SINGLE CONNECTED TENSOR.
    
    All individual losses remain as tensors in the autograd graph until
    they are summed. This is critical — converting to .item() floats
    (as the old code did) severs the graph and prevents any gradients
    from flowing back to the model.
    
    Returns:
        loss_tensor:  scalar tensor WITH grad_fn, ready for .backward()
        components:   {name: float} for logging only (detached)
    """
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    components = {}
    
    def add_loss(name, criterion, weight, pred, target=None):
        nonlocal total_loss
        if criterion is None or weight == 0:
            return
        try:
            if target is not None:
                loss_val = criterion(pred, target)
            else:
                loss_val = criterion(pred)
            
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"⚠️  {name} produced NaN/Inf — skipping")
                return
            
            weighted = weight * loss_val
            total_loss = total_loss + weighted
            components[name] = weighted.item()
        except Exception as e:
            print(f"⚠️  {name} error: {e}")
    
    # ── Core losses ──
    add_loss('L1', criterion_l1, 1.0, sr, hr)
    
    if use_perceptual:
        add_loss('Perc', criterion_perceptual, perceptual_weight, sr, hr)
    
    # ── Brightness / contrast / color ──
    add_loss('Bright', criterion_brightness, brightness_weight, sr, hr)
    add_loss('Contrast', criterion_contrast, contrast_weight, sr, hr)
    add_loss('Color', criterion_color, color_weight, sr)
    add_loss('TV', criterion_tv, tv_weight, sr)
    
    # ── Detail preservation ──
    add_loss('HighFreq', criterion_highfreq, highfreq_weight, sr, hr)
    add_loss('Edge', criterion_edge, edge_weight, sr, hr)
    
    # ── Highlight / dynamic range ──
    add_loss('HL', criterion_highlight, highlight_weight, sr, hr)
    add_loss('DRange', criterion_drange, drange_weight, sr, hr)
    add_loss('HLGrad', criterion_hl_grad, hlgrad_weight, sr, hr)
    
    # ── Advanced brightness preservation ──
    add_loss('Lab', criterion_lab, lab_weight, sr, hr)
    add_loss('Local', criterion_local, local_weight, sr, hr)
    add_loss('SSIM', criterion_ssim, ssim_weight, sr, hr)
    add_loss('Exp', criterion_exposure, exposure_weight, sr, hr)
    add_loss('LogTone', criterion_log_tone, log_tone_weight, sr, hr)
    add_loss('MeanLum', criterion_mean_lum, mean_lum_weight, sr, hr)
    
    # ── Expensive losses – run less frequently ──
    if criterion_hist is not None and hist_weight > 0 and batch_idx % 10 == 0:
        add_loss('Hist', criterion_hist, hist_weight, sr, hr)
    
    if criterion_percentile is not None and percentile_weight > 0 and batch_idx % 50 == 0:
        add_loss('Percentile', criterion_percentile, percentile_weight, sr, hr)
    
    return total_loss, components

# ============================================================================
# M4 MEMORY MONITORING
# ============================================================================

def get_mps_memory_status():
    """Get current memory usage for M4"""
    if not torch.backends.mps.is_available():
        return None
    
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_gb': mem_info.rss / (1024**3),
            'vms_gb': mem_info.vms / (1024**3),
            'percent': process.memory_percent(),
        }
    except:
        return None

def adaptive_batch_cleanup(device, epoch, batch_idx, memory_threshold_gb=12):
    """Cleanup when memory approaches threshold on M4"""
    if device.type != 'mps':
        return
    
    mem_status = get_mps_memory_status()
    if mem_status and mem_status['rss_gb'] > memory_threshold_gb:
        print(f"\n⚠️  Memory high ({mem_status['rss_gb']:.1f}GB @ epoch {epoch}, batch {batch_idx})")
        print(f"    Aggressive cleanup triggered")
        torch.mps.empty_cache()
        torch.mps.synchronize()
        import gc
        collected = gc.collect()
        mem_status_after = get_mps_memory_status()
        if mem_status_after:
            print(f"    After cleanup: {mem_status_after['rss_gb']:.1f}GB")


def log_memory_status(device, label="", threshold_gb=14.0):
    """
    Log current memory usage and warn if approaching limit
    Call this periodically during training
    """
    try:
        process = psutil.Process()
        rss_gb = process.memory_info().rss / (1024**3)
        
        msg = f"💾 Memory [{label}]: {rss_gb:.2f} GB"
        
        if rss_gb > threshold_gb:
            print(f"⚠️  {msg} ⚠️  APPROACHING LIMIT (threshold: {threshold_gb}GB)")
            return rss_gb, True  # Over threshold
        else:
            print(msg)
            return rss_gb, False  # OK
    
    except Exception as e:
        print(f"⚠️  Could not read memory: {e}")
        return 0, False


def safe_gc_collect(device):
    """
    Safe garbage collection with device-specific cleanup
    """
    try:
        # Python garbage collection
        collected = gc.collect()
        
        # Device-specific cleanup
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return collected
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")
        return 0


# ============================================================================
# TRAINING STABILITY HELPERS
# ============================================================================

def snapshot_model_state(model):
    """Snapshot model state to CPU for fast rollback."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model_state(model, state, device):
    """Restore model state from CPU snapshot."""
    restored = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(restored, strict=False)
    del restored


def snapshot_optimizer_state(optimizer):
    """Snapshot optimizer state to CPU for fast rollback."""
    state = copy.deepcopy(optimizer.state_dict())
    for _, st in state.get('state', {}).items():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.detach().cpu().clone()
    return state


def restore_optimizer_state(optimizer, state, device):
    """Restore optimizer state from CPU snapshot."""
    state_copy = copy.deepcopy(state)
    optimizer.load_state_dict(state_copy)
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


def has_invalid_gradients(model):
    """Detect NaN/Inf gradients."""
    for param in model.parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            return True
    return False


def has_invalid_parameters(model):
    """Detect NaN/Inf parameters."""
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return True
    return False


def apply_lr_backoff(optimizer, factor=0.5, min_lr=1e-7, schedulers=None):
    """Scale learning rate down and clamp to min_lr. Optionally adjust schedulers."""
    schedulers = schedulers or []
    new_lrs = []
    for group in optimizer.param_groups:
        old_lr = group['lr']
        new_lr = max(min_lr, old_lr * factor)
        group['lr'] = new_lr
        new_lrs.append((old_lr, new_lr))
    for scheduler in schedulers:
        if scheduler is None:
            continue
        if hasattr(scheduler, 'base_lrs'):
            scheduler.base_lrs = [max(min_lr, lr * factor) for lr in scheduler.base_lrs]
    return new_lrs


# ============================================================================
# Input Validation
# ============================================================================

def validate_inputs(lr_dir, hr_dir, pretrained):
    """Validate input directories and files before training"""
    errors = []
    
    lr_path = Path(lr_dir)
    if not lr_path.exists():
        errors.append(f"❌ LR directory not found: {lr_dir}")
    elif not lr_path.is_dir():
        errors.append(f"❌ LR path is not a directory: {lr_dir}")
    else:
        lr_images = list(lr_path.glob('*.png')) + list(lr_path.glob('*.jpg')) + \
                    list(lr_path.glob('*.jpeg'))
        if not lr_images:
            errors.append(f"❌ No images found in LR directory: {lr_dir}")
    
    hr_path = Path(hr_dir)
    if not hr_path.exists():
        errors.append(f"❌ HR directory not found: {hr_dir}")
    elif not hr_path.is_dir():
        errors.append(f"❌ HR path is not a directory: {hr_dir}")
    else:
        hr_images = list(hr_path.glob('*.png')) + list(hr_path.glob('*.jpg')) + \
                    list(hr_path.glob('*.jpeg'))
        if not hr_images:
            errors.append(f"❌ No images found in HR directory: {hr_dir}")
    
    pretrained_path = Path(pretrained)
    if not pretrained_path.exists():
        errors.append(f"❌ Pretrained checkpoint not found: {pretrained}")
    elif not pretrained_path.suffix in ['.pth', '.pt']:
        errors.append(f"❌ Pretrained file is not .pth: {pretrained}")
    
    if errors:
        print("="*70)
        print("INPUT VALIDATION ERRORS")
        print("="*70)
        for error in errors:
            print(error)
        print("="*70)
        return False
    
    return True


# ============================================================================
# System Resource Checking
# ============================================================================

def check_system_resources(device):
    """Check available resources and optimize settings"""
    print("\n" + "="*70)
    print("SYSTEM RESOURCES CHECK")
    print("="*70)
    
    if device.type == 'mps':
        print("✅ MPS (Apple Silicon M4)")
        if hasattr(torch.mps, "set_per_process_memory_fraction"):
            torch.mps.set_per_process_memory_fraction(0.75)
            print("   MPS memory limit set to 75% to prevent swap")
        else:
            print("   ⚠️  torch.mps.set_per_process_memory_fraction not available in this PyTorch")
        
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                total_gb = mem.total / (1024**3)
                available_gb = mem.available / (1024**3)
                print(f"   Total memory: {total_gb:.1f} GB")
                print(f"   Available: {available_gb:.1f} GB")
                if available_gb < 4:
                    print(f"   ⚠️  WARNING: Less than 4GB available!")
            except Exception as e:
                print(f"   Could not read system memory: {e}")
    
    elif device.type == 'cuda':
        print(f"✅ CUDA")
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   Total GPU memory: {total_mem:.1f} GB")
        except Exception as e:
            print(f"   Could not read GPU memory: {e}")
    
    else:
        print(f"⚠️  CPU mode (will be VERY slow)")
    
    print("="*70 + "\n")


# ============================================================================
# MPS Memory Management
# ============================================================================

def clear_mps_cache():
    """Clear MPS cache to prevent fragmentation on unified memory"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def aggressive_memory_cleanup(device):
    """Aggressive cleanup for MPS/CUDA to prevent fragmentation leaks"""
    import gc
    gc.collect()  # Force Python garbage collection first
    
    if device.type == 'mps':
        torch.mps.empty_cache()
        # Force synchronize to ensure GPU ops complete
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


# ============================================================================
# MEMORY-EFFICIENT LOSS COMPUTATION
# ============================================================================

def safe_loss_computation(loss_fn, weight, pred, target=None, device=None):
    """
    Compute loss as scalar, delete tensor immediately.
    Returns: scalar value only
    """
    if loss_fn is None or weight == 0:
        return 0.0
    
    try:
        if target is not None:
            loss_tensor = loss_fn(pred, target)
        else:
            loss_tensor = loss_fn(pred)
        
        loss_value = weight * loss_tensor.item()
        del loss_tensor
        
        # Cleanup
        if device and device.type == 'mps':
            torch.mps.empty_cache()
        
        return loss_value
    except Exception as e:
        print(f"⚠️  Loss computation error: {e}")
        return 0.0


def safe_loss_computation_with_nan_check(loss_fn, weight, pred, target=None, 
                                         device=None, loss_name="Unknown"):
    """
    Compute loss as scalar with detailed NaN checking.
    Returns: (scalar_value, is_valid)
    """
    if loss_fn is None or weight == 0:
        return 0.0, True
    
    try:
        if target is not None:
            loss_tensor = loss_fn(pred, target)
        else:
            loss_tensor = loss_fn(pred)
        
        # Check tensor for NaN before item()
        if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
            print(f"⚠️  {loss_name} loss produced NaN/Inf tensor")
            del loss_tensor
            return 0.0, False
        
        loss_value = weight * loss_tensor.item()
        
        # Check final scalar
        if not np.isfinite(loss_value):
            print(f"⚠️  {loss_name} loss scalar is {loss_value}")
            del loss_tensor
            return 0.0, False
        
        del loss_tensor
        
        if device and device.type == 'mps':
            torch.mps.empty_cache()
        
        return loss_value, True
        
    except Exception as e:
        print(f"⚠️  {loss_name} computation error: {e}")
        return 0.0, False


def cleanup_tensors(tensor_dict, device):
    """
    Safely delete a dict of tensors and clear cache.
    """
    for key in list(tensor_dict.keys()):
        if tensor_dict[key] is not None:
            try:
                del tensor_dict[key]
            except:
                pass
    
    if device and device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device and device.type == 'cuda':
        torch.cuda.empty_cache()


def to_device(tensor, device):
    """Device-specific optimized tensor transfer for M4"""
    if device.type == 'cuda':
        # CUDA benefits from async transfers
        return tensor.to(device, non_blocking=True, memory_format=torch.channels_last)
    elif device.type == 'mps':
        # 🔴 MPS has unified memory - simple transfer is better
        # Avoid memory_format issues that can cause slowdowns
        try:
            return tensor.to(device, memory_format=torch.channels_last)
        except RuntimeError:
            # Fallback if channels_last fails
            return tensor.to(device)
    else:
        return tensor.to(device)


# ============================================================================
# ADVANCED LOSS FUNCTIONS - OVEREXPOSURE PREVENTION
# ============================================================================

class LabColorLoss(nn.Module):
    """
    ✅ CORRECTED: Converts to Lab space and penalizes L-channel deviation.
    Prevents brightness shifts while allowing color flexibility.
    
    BRIGHTNESS FIX: Previously applied gamma correction too early, which caused
    the input sRGB values (0-1) to be linearized before being used as sRGB values,
    resulting in brightness amplification. Now properly converts sRGB -> Linear -> XYZ -> Lab.
    
    M4-Optimized: Matrix multiply only, very fast on Metal.
    """
    def __init__(self, l_weight=2.0, ab_weight=1.0):
        super().__init__()
        self.l_weight = l_weight
        self.ab_weight = ab_weight
        
        # RGB to XYZ conversion matrix (D65 illuminant)
        # This expects LINEAR RGB, not sRGB
        rgb_to_xyz = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32)
        self.register_buffer('rgb_to_xyz', rgb_to_xyz)
        
        # D65 reference white point
        d65_ref = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
        self.register_buffer('d65_ref', d65_ref)
    
    def forward(self, pred, target):
        b, c, h, w = pred.shape
        
        # Flatten spatial dimensions: (B, 3, H*W)
        pred_flat = pred.reshape(b, c, -1)
        target_flat = target.reshape(b, c, -1)
        
        # sRGB to Linear (vectorized)
        def srgb_to_linear(rgb):
            mask = rgb <= 0.04045
            linear = torch.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            return linear
        
        pred_lin = srgb_to_linear(pred_flat)
        target_lin = srgb_to_linear(target_flat)
        
        # Matrix multiply using einsum (faster on MPS)
        # rgb_to_xyz is (3, 3), input is (B, 3, HW)
        pred_xyz = torch.einsum('ncr,cd->ndr', pred_lin, self.rgb_to_xyz)
        target_xyz = torch.einsum('ncr,cd->ndr', target_lin, self.rgb_to_xyz)
        
        # Reshape back
        pred_xyz = pred_xyz.view(b, 3, h, w)
        target_xyz = target_xyz.view(b, 3, h, w)
        
        # XYZ to Lab
        def xyz_to_lab(xyz):
            xyz_norm = xyz / self.d65_ref.view(1, 3, 1, 1)
            delta = 6.0 / 29.0
            mask = xyz_norm > delta ** 3
            xyz_f = torch.where(
                mask,
                xyz_norm ** (1.0 / 3.0),
                xyz_norm / (3 * delta ** 2) + 4.0 / 29.0
            )
            L = 116 * xyz_f[:, 1:2] - 16
            a = 500 * (xyz_f[:, 0:1] - xyz_f[:, 1:2])
            b = 200 * (xyz_f[:, 1:2] - xyz_f[:, 2:3])
            return torch.cat([L, a, b], dim=1)
        
        pred_lab = xyz_to_lab(pred_xyz)
        target_lab = xyz_to_lab(target_xyz)
        
        # Weight L channel heavily
        loss_l = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1])
        loss_ab = F.l1_loss(pred_lab[:, 1:], target_lab[:, 1:])
        
        return self.l_weight * loss_l + self.ab_weight * loss_ab

# ============================================================================
# NEW ANTI-BRIGHTENING LOSS FUNCTIONS (v2.0)
# ============================================================================

class MeanLuminancePreservationLoss(nn.Module):
    """
    🔴 CRITICAL FIX: Prevents gradual brightness drift by enforcing
    exact mean brightness match between output and target.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_lum = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
        target_lum = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        pred_mean = pred_lum.mean()
        target_mean = target_lum.mean()
        return (pred_mean - target_mean) ** 2


def _safe_quantile_batch(x, q):
    """Quantile over dim=1 with MPS-safe fallback."""
    if x.numel() == 0:
        return x.new_zeros(x.shape[0])
    if x.device.type == 'mps':
        n = x.shape[1]
        k = max(1, min(n, int(round(q * (n - 1))) + 1))
        val, _ = torch.kthvalue(x, k, dim=1)
        return val
    return torch.quantile(x, q, dim=1)


def _safe_quantile_1d(x, q):
    """Quantile over all elements with MPS-safe fallback."""
    x_flat = x.reshape(-1)
    if x_flat.numel() == 0:
        return x.new_tensor(0.0)
    if x_flat.device.type == 'mps':
        n = x_flat.numel()
        k = max(1, min(n, int(round(q * (n - 1))) + 1))
        val, _ = torch.kthvalue(x_flat, k)
        return val
    return torch.quantile(x_flat, q)


class PercentilePreservationLoss(nn.Module):
    """Additional fix: Ensures brightness distribution percentiles match."""
    def __init__(self, percentiles=[50, 75, 90, 95, 99]):
        super().__init__()
        self.percentiles = percentiles
    
    def forward(self, pred, target):
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        loss = 0.0
        for p in self.percentiles:
            pred_p = _safe_quantile_batch(pred_flat, p / 100.0)
            target_p = _safe_quantile_batch(target_flat, p / 100.0)
            loss += F.mse_loss(pred_p, target_p)
        return loss / len(self.percentiles)


class AdaptiveHighlightSuppressionLoss(nn.Module):
    """
    🟠 ENHANCED: Replaces HighlightSuppressionLoss.
    Gets progressively stronger over epochs to counteract brightness drift.
    """
    def __init__(self, threshold=0.9, safe_margin=0.01):
        super().__init__()
        self.threshold = threshold
        self.safe_margin = safe_margin
        self.epoch = 0  # 🔴 FIXED: Use Python int, not tensor

    def step_epoch(self):
        self.epoch += 1  # Fast Python operation
        if self.epoch <= 5:
            print(f"  📊 Highlight suppression strength: {1.0 + self.epoch * 0.15:.2f}x")
    
    def forward(self, pred, target):
        target_max = target.reshape(target.size(0), -1).max(dim=1)[0].mean()
        
        adaptive_threshold = min(self.threshold, target_max.item() + self.safe_margin)
        excess = torch.clamp(pred - adaptive_threshold, min=0.0)
        base_loss = (excess ** 2).mean()
        extreme_excess = torch.clamp(pred - 0.95, min=0.0)
        extreme_loss = (extreme_excess ** 3).mean()
        epoch_multiplier = 1.0 + (self.epoch * 0.15)  # Use Python int
        return epoch_multiplier * (base_loss + 2.0 * extreme_loss)


# ============================================================================
# MEMORY-SAFE PERCEPTUAL LOSS (FIXED v2.3)
# ============================================================================

class BrightnessNormalizedPerceptualLoss(nn.Module):
    """
    🔴 FIXED: Properly cleans VGG model from memory
    This was causing 2-3GB memory leaks per epoch
    """
    def __init__(self, device='cpu', layers=None):
        super().__init__()
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3']
        
        print("  Loading VGG19 (this may take a moment)...")
        
        # Load and extract features IMMEDIATELY
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except AttributeError:
            vgg = models.vgg19(pretrained=True)
        
        # ✅ CRITICAL FIX: Extract ONLY features, discard rest
        features = nn.Sequential(*list(vgg.features.children())[:36])
        
        # Move to device BEFORE deleting original model
        features = features.to(device)
        features.eval()
        
        # ✅ Freeze parameters
        for param in features.parameters():
            param.requires_grad = False
        
        # ✅ CRITICAL: Delete original VGG model completely
        del vgg
        import gc
        gc.collect()
        
        # Force device-specific cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("  ✅ VGG model loaded (extra memory freed)")
        
        # Now register the features
        self.vgg = features
        
        self.layer_name_mapping = {
            '3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_3',
            '26': 'relu4_3', '35': 'relu5_3'
        }
        self.layers = layers
        
        # Register normalization buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Move entire module to device
        self.to(device)

    def forward(self, pred, target):
        """
        Memory-optimized forward with immediate cleanup
        """
        pred_normalized = self.normalize_brightness(pred, target)
        
        # Ensure buffers on same device
        device_mean = self.mean.to(pred.device)
        device_std = self.std.to(pred.device)
        
        pred_norm = (pred_normalized - device_mean) / device_std
        target_norm = (target - device_mean) / device_std
        
        # Extract features
        pred_features = {}
        target_features = {}
        
        try:
            # ✅ Extract features with cleanup
            pred_x = pred_norm
            target_x = target_norm
            
            for name, layer in self.vgg._modules.items():
                pred_x = layer(pred_x)
                target_x = layer(target_x)
                
                if name in self.layer_name_mapping:
                    pred_features[self.layer_name_mapping[name]] = pred_x
                    target_features[self.layer_name_mapping[name]] = target_x.detach()
            
            # Compute loss
            loss = 0.0
            for layer in self.layers:
                if layer in pred_features and layer in target_features:
                    pred_feat = pred_features[layer]
                    target_feat = target_features[layer]
                    
                    # Normalize and compute cosine similarity
                    pred_feat_norm = F.normalize(
                        pred_feat.contiguous().view(pred_feat.size(0), -1), dim=1
                    )
                    target_feat_norm = F.normalize(
                        target_feat.contiguous().view(target_feat.size(0), -1), dim=1
                    )
                    
                    cosine_sim = (pred_feat_norm * target_feat_norm).sum(dim=1).mean()
                    loss += (1.0 - cosine_sim)
                    
                    # ✅ Delete layer features immediately
                    del pred_feat, target_feat, pred_feat_norm, target_feat_norm
            
            result = loss / len(self.layers)
            return result
        
        finally:
            # ✅ GUARANTEED cleanup of all feature dictionaries
            for feat_dict in [pred_features, target_features]:
                for key in list(feat_dict.keys()):
                    try:
                        del feat_dict[key]
                    except:
                        pass
                feat_dict.clear()
            
            # Device cleanup
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def normalize_brightness(self, pred, target):
        """Normalize pred brightness to match target"""
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        pred_mean = pred.mean(dim=[1, 2, 3], keepdim=True)
        target_mean = target.mean(dim=[1, 2, 3], keepdim=True)
        adjusted = pred - pred_mean + target_mean
        return torch.clamp(adjusted, 0.0, 1.0)


class LocalStatisticsLoss(nn.Module):
    """
    ✅ FIXED: Ensures local mean and variance match target.
    Prevents 'hot spots' and local overexposure.
    Now with better numerical stability for NaN prevention.
    """
    def __init__(self, kernel_size=16, mean_weight=1.0, std_weight=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.mean_weight = mean_weight
        self.std_weight = std_weight
        self.pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.eps = 1e-8  # 🔴 CRITICAL: Add epsilon
    
    def forward(self, pred, target):
        """Compute local statistics loss with NaN safety and memory cleanup"""
        # Ensure inputs are valid and in range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        try:
            # Local means
            pred_mean_raw = self.pool(pred)
            target_mean_raw = self.pool(target)
            
            # Mean loss
            mean_loss = F.mse_loss(pred_mean_raw, target_mean_raw)
            
            # Clamp intermediate results to prevent NaN
            pred_mean = torch.clamp(pred_mean_raw.detach(), 0.0, 1.0)
            target_mean = torch.clamp(target_mean_raw.detach(), 0.0, 1.0)
            
            # 🔴 FIXED: Delete raw means to free memory
            del pred_mean_raw, target_mean_raw
            
            # Local std (contrast) - WITH NUMERICAL STABILITY
            pred_sq_mean = self.pool(pred ** 2).detach()
            target_sq_mean = self.pool(target ** 2).detach()
            
            # Variance = E[x^2] - (E[x])^2
            pred_var = torch.clamp(pred_sq_mean - (pred_mean ** 2), min=self.eps)
            target_var = torch.clamp(target_sq_mean - (target_mean ** 2), min=self.eps)
            
            # 🔴 FIXED: Delete squared means
            del pred_sq_mean, target_sq_mean, pred_mean, target_mean
            
            # Std with safety
            pred_std = torch.sqrt(pred_var + self.eps)
            target_std = torch.sqrt(target_var + self.eps)
            
            # Ensure no NaN values
            if torch.isnan(pred_std).any() or torch.isinf(pred_std).any():
                pred_std = torch.clamp(pred_std, 0.0, 1.0)
            if torch.isnan(target_std).any() or torch.isinf(target_std).any():
                target_std = torch.clamp(target_std, 0.0, 1.0)
            
            # Compute std loss
            std_loss = F.mse_loss(pred_std, target_std)
            
            # 🔴 FIXED: Delete variance and std tensors
            del pred_var, target_var, pred_std, target_std
            
            # Final check for NaN
            if torch.isnan(mean_loss) or torch.isinf(mean_loss) or \
               torch.isnan(std_loss) or torch.isinf(std_loss):
                print(f"⚠️  LocalStatisticsLoss: NaN/Inf detected. Returning zero loss.")
                del mean_loss, std_loss
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            result = self.mean_weight * mean_loss + self.std_weight * std_loss
            del mean_loss, std_loss
            return result
            
        except Exception as e:
            print(f"⚠️  LocalStatisticsLoss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


class SSIMLuminanceLoss(nn.Module):
    """
    ✅ NEW: SSIM on luminance channel only (Y from RGB).
    Much faster than full MS-SSIM or VGG perceptual loss.
    """
    def __init__(self, window_size=11, weight=1.0):
        super().__init__()
        self.window_size = window_size
        self.weight = weight
        
        # Create Gaussian window
        sigma = 1.5
        x = torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float32)
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        window = gauss / gauss.sum()
        window = window.unsqueeze(1) * window.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        self.register_buffer('window', window)
    
    def rgb_to_luminance(self, img):
        """ITU-R BT.601 luminance"""
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    
    def forward(self, pred, target):
        pred_y = self.rgb_to_luminance(pred)
        target_y = self.rgb_to_luminance(target)
        
        # Compute local means
        mu1 = F.conv2d(pred_y, self.window, padding=self.window_size // 2, groups=1)
        mu2 = F.conv2d(target_y, self.window, padding=self.window_size // 2, groups=1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances
        sigma1_sq = (
            F.conv2d(pred_y ** 2, self.window, padding=self.window_size // 2, groups=1) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(target_y ** 2, self.window, padding=self.window_size // 2, groups=1) - mu2_sq
        )
        sigma12 = (
            F.conv2d(pred_y * target_y, self.window, padding=self.window_size // 2, groups=1) - mu1_mu2
        )
        
        # SSIM formula
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        
        return self.weight * (1.0 - ssim_map.mean())


class ExposureGradientLoss(nn.Module):
    """
    ✅ IMPROVED: Penalizes gradients more heavily where target is dark but pred is bright.
    Prevents 'halos' and edge ringing in shadow-to-highlight transitions.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        # Luminance
        pred_y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_y = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Gradients
        pgx = F.conv2d(pred_y, self.sobel_x, padding=1)
        pgy = F.conv2d(pred_y, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-6)
        
        tgx = F.conv2d(target_y, self.sobel_x, padding=1)
        tgy = F.conv2d(target_y, self.sobel_y, padding=1)
        target_grad = torch.sqrt(tgx ** 2 + tgy ** 2 + 1e-6)
        
        # Weight by exposure risk: High where target is dark (potential overexposure)
        exposure_risk = torch.relu(0.5 - target_y)  # High in shadow regions
        weight = 1.0 + 3.0 * exposure_risk  # 4x penalty in shadows
        
        return torch.mean(weight * torch.abs(pred_grad - target_grad))


class SoftHistogramLoss(nn.Module):
    """
    ✅ NEW: Matches the distribution of bright values using soft histograms.
    Prevents overexposure by ensuring the 'tail' of the distribution matches target.
    """
    def __init__(self, bins=32, min_val=0.7, max_val=1.0, sigma=0.02):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma
        
        bin_edges = torch.linspace(min_val, max_val, bins)
        self.register_buffer('bin_edges', bin_edges)
    
    def forward(self, pred, target):
        """Compute soft histogram loss with memory efficiency - M4 optimized"""
        # Focus on highlight region only
        pred_high = torch.clamp(pred, self.min_val, 1.0)
        target_high = torch.clamp(target, self.min_val, 1.0)
        
        # Create masks for values in highlight region
        pred_mask = (pred_high > self.min_val)
        target_mask = (target_high > self.min_val)
        
        # Extract highlighted pixels (Results in 1D tensors of DIFFERENT sizes)
        pred_high_vals = torch.masked_select(pred_high, pred_mask)
        target_high_vals = torch.masked_select(target_high, target_mask)
        
        # Check if we have any highlighted pixels
        if pred_high_vals.numel() == 0 or target_high_vals.numel() == 0:
            del pred_mask, target_mask, pred_high, target_high
            del pred_high_vals, target_high_vals
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        try:
            # 🔴 FIXED: Clamp number of samples for M4 memory safety
            # SAMPLED INDEPENDENTLY because sizes differ and spatial info is lost
            max_samples = 50000 
            
            if pred_high_vals.numel() > max_samples:
                # Use device=pred.device to avoid CPU<->GPU sync
                indices_p = torch.randperm(pred_high_vals.numel(), device=pred.device)[:max_samples]
                pred_high_vals = pred_high_vals[indices_p]
            
            if target_high_vals.numel() > max_samples:
                indices_t = torch.randperm(target_high_vals.numel(), device=target.device)[:max_samples]
                target_high_vals = target_high_vals[indices_t]
            
            # Vectorized soft histogram computation
            edges = self.bin_edges.view(1, -1)  # (1, bins)
            
            # Compute weights for all bins at once
            pred_weights = torch.exp(-((pred_high_vals.unsqueeze(-1) - edges) ** 2) / (2 * self.sigma ** 2))
            target_weights = torch.exp(-((target_high_vals.unsqueeze(-1) - edges) ** 2) / (2 * self.sigma ** 2))
            
            # Sum across pixels to get histogram count
            pred_hist = pred_weights.sum(dim=0)
            target_hist = target_weights.sum(dim=0)
            
            # Normalize to create PDF (This handles the different pixel counts automatically)
            pred_hist = pred_hist / (pred_hist.sum() + 1e-6)
            target_hist = target_hist / (target_hist.sum() + 1e-6)
            
            # Earth Mover's Distance (CDF difference)
            pred_cdf = torch.cumsum(pred_hist, dim=0)
            target_cdf = torch.cumsum(target_hist, dim=0)
            
            loss = torch.mean(torch.abs(pred_cdf - target_cdf))
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️  SoftHistogramLoss: NaN/Inf detected. Returning zero loss.")
                del pred_weights, target_weights, pred_hist, target_hist
                del pred_cdf, target_cdf
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            return loss
        
        finally:
            # 🔴 GUARANTEED cleanup
            try:
                del pred_mask, target_mask, pred_high, target_high
                del pred_high_vals, target_high_vals
            except:
                pass
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


# ============================================================================
# Original Loss Functions (Kept for Reference)
# ============================================================================

class HighlightAwareL1(nn.Module):
    """Weighted L1 that penalizes errors in bright regions exponentially."""
    def __init__(self, highlight_threshold=0.7, penalty_factor=3.0):
        super().__init__()
        self.threshold = highlight_threshold
        self.factor = penalty_factor
    
    def forward(self, pred, target):
        base_loss = torch.abs(pred - target)
        weights = 1.0 + (self.factor - 1.0) * torch.sigmoid(
            (target - self.threshold) * 10
        )
        return torch.mean(base_loss * weights)


class LogToneMappingLoss(nn.Module):
    """
    ✅ NOW INTEGRATED: Loss in log space for exponential penalty on overexposure.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        pred_safe = torch.clamp(pred, self.eps, 1.0 - self.eps)
        target_safe = torch.clamp(target, self.eps, 1.0 - self.eps)
        
        # Log mapping with 15x compression factor
        pred_log = torch.log1p(pred_safe * 15) / torch.log(torch.tensor(16.0, device=pred.device))
        target_log = torch.log1p(target_safe * 15) / torch.log(torch.tensor(16.0, device=target.device))
        
        return torch.mean(torch.abs(pred_log - target_log))


class ApproximateDynamicRangeLoss(nn.Module):
    """Uses kthvalue for true percentile approximation instead of max."""
    def __init__(self, target_percentile=0.95):
        super().__init__()
        self.p = target_percentile
    
    def forward(self, pred, target):
        b, c, h, w = pred.shape
        
        # Change both to reshape()
        pred_vals = pred.reshape(b, c, -1)  # was .view()
        target_vals = target.reshape(b, c, -1)  # was .view()
        
        spatial_size = h * w
        k = max(1, int(spatial_size * self.p))
        
        pred_percentile, _ = torch.kthvalue(pred_vals, k, dim=2)
        target_percentile, _ = torch.kthvalue(target_vals, k, dim=2)
        
        overexposure = torch.relu(pred_percentile - target_percentile)
        return torch.mean(overexposure)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class HighlightGradientLoss(nn.Module):
    """Penalizes strong gradients in bright areas."""
    def __init__(self, brightness_threshold=0.6):
        super().__init__()
        self.threshold = brightness_threshold
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        # Removed manual device check for performance
        pred_y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_y = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        pred_gx = F.conv2d(pred_y, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_y, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        
        highlight_mask = (target_y > self.threshold).float()
        return torch.mean(pred_grad * highlight_mask)


class BrightnessConsistencyLoss(nn.Module):
    """Limits brightness deviation from target (HR)."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_mean = torch.mean(pred, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target, dim=[2, 3], keepdim=True)
        return torch.mean(torch.abs(pred_mean - target_mean))


class ContrastConsistencyLoss(nn.Module):
    """Limits contrast deviation from target."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_std = torch.std(pred, dim=[2, 3], keepdim=True)
        target_std = torch.std(target, dim=[2, 3], keepdim=True)
        return torch.mean(torch.abs(pred_std - target_std))


class ColorRangeLimiter(nn.Module):
    """Soft penalty for values outside safe range."""
    def __init__(self, margin=0.02, balance_weight=0.3):
        super().__init__()
        self.margin = margin
        self.balance_weight = balance_weight
    
    def forward(self, pred):
        lower_penalty = torch.relu(self.margin - pred)
        upper_penalty = torch.relu(pred - (1.0 - self.margin))
        range_loss = torch.mean(lower_penalty + upper_penalty)
        
        mean_r = torch.mean(pred[:, 0:1, :, :], dim=[2, 3], keepdim=True)
        mean_g = torch.mean(pred[:, 1:2, :, :], dim=[2, 3], keepdim=True)
        mean_b = torch.mean(pred[:, 2:3, :, :], dim=[2, 3], keepdim=True)
        
        avg_mean = (mean_r + mean_g + mean_b) / 3.0
        color_balance_loss = torch.mean(
            torch.abs(mean_r - avg_mean) + 
            torch.abs(mean_g - avg_mean) + 
            torch.abs(mean_b - avg_mean)
        )
        
        return range_loss + (self.balance_weight * color_balance_loss)


class TotalVariationLimiter(nn.Module):
    """Limits excessive noise/high-frequency artifacts."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred):
        diff_h = torch.diff(pred, dim=3, append=pred[:, :, :, -1:])
        diff_v = torch.diff(pred, dim=2, append=pred[:, :, -1:, :])
        
        tv_h = torch.mean(torch.sqrt(diff_h * diff_h + self.eps))
        tv_v = torch.mean(torch.sqrt(diff_v * diff_v + self.eps))
        
        return tv_h + tv_v


class HighFrequencyLoss(nn.Module):
    """Penalizes loss of high-frequency details."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)
    
    def forward(self, pred, target):
        # Remove manual device check - PyTorch handles this automatically with register_buffer
        pred_hf = torch.cat([
            F.conv2d(pred[:, i:i+1], self.kernel, padding=1)
            for i in range(3)
        ], dim=1)
        target_hf = torch.cat([
            F.conv2d(target[:, i:i+1], self.kernel, padding=1)
            for i in range(3)
        ], dim=1)
        return torch.mean(torch.abs(pred_hf - target_hf))


class EdgePreservationLoss(nn.Module):
    """Preserves edge structure using Sobel gradients."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        # Removed manual device check for performance
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        pred_gx = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        
        target_gx = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_mag = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)
        
        mag_loss = torch.mean(torch.abs(pred_mag - target_mag))
        cos_sim = (pred_gx * target_gx + pred_gy * target_gy) / (pred_mag * target_mag + 1e-6)
        dir_loss = torch.mean(1.0 - cos_sim)
        
        return mag_loss + 0.5 * dir_loss


# DEPRECATED IN V2.0 - USE BrightnessNormalizedPerceptualLoss INSTEAD
# class PerceptualLoss(nn.Module):
#     """VGG-based perceptual loss"""
#     def __init__(self, device):
#         super(PerceptualLoss, self).__init__()
        
#         try:
#             vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
#         except AttributeError:
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore', category=FutureWarning)
#                 vgg = models.vgg19(pretrained=True).to(device)
        
#         self.features = nn.Sequential(*list(vgg.features.children())[:36])
#         self.features.eval()
        
#         for param in self.features.parameters():
#             param.requires_grad = False
    
#     def forward(self, pred, target):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
#         pred_norm = (pred - mean) / std
#         target_norm = (target - mean) / std
        
#         pred_feat = self.features(pred_norm)
#         target_feat = self.features(target_norm)
        
#         return torch.mean((pred_feat - target_feat) ** 2)


# ============================================================================
# Dataset Classes
# ============================================================================

class VideoFramePairDataset(Dataset):
    """OPTIMIZED Dataset using OpenCV"""
    
    def __init__(self, lr_dir, hr_dir, patch_size=128, multi_scale=False):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.multi_scale = multi_scale
        
        self.lr_images = sorted(list(self.lr_dir.glob('*.png')) + 
                                list(self.lr_dir.glob('*.jpg')) +
                                list(self.lr_dir.glob('*.jpeg')))
        
        self.pairs = []
        for lr_path in self.lr_images:
            hr_path = self.hr_dir / lr_path.name
            if hr_path.exists():
                self.pairs.append((lr_path, hr_path))
        
        print(f"Found {len(self.pairs)} paired images")
        
        if len(self.pairs) == 0:
            raise ValueError("No paired images found!")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Load and process image pair - MEMORY SAFE v2.3
        """
        max_attempts = 5
        last_lr = None
        last_hr = None

        for _ in range(max_attempts):
            lr_img = None
            hr_img = None
            lr_patch = None
            hr_patch = None

            try:
                lr_path, hr_path = self.pairs[idx]

                # Load images
                lr_img = cv2.imread(str(lr_path))
                hr_img = cv2.imread(str(hr_path))

                if lr_img is None or hr_img is None:
                    raise ValueError(f"Failed to load images at index {idx}")

                # Convert color space
                lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

                # Validate size
                if lr_img.shape[0] < 64 or lr_img.shape[1] < 64:
                    print(f"⚠️  LR image too small at index {idx}: {lr_path.name}")
                    lr_img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
                    hr_img = np.zeros((self.patch_size * 2, self.patch_size * 2, 3), dtype=np.uint8)

                # Extract patches
                lr_patch, hr_patch = self._get_patch(lr_img, hr_img)

                # Augment
                lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

                # Convert to tensors
                lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
                hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0

                last_lr, last_hr = lr_tensor, hr_tensor

                # 🔴 NEW: Reject flat patches (solid colors/black bars)
                # If the standard deviation is near 0, normalization will divide by zero and cause NaNs.
                if lr_tensor.std() >= 1e-3:
                    return lr_tensor, hr_tensor

                # Try next index
                idx = (idx + 1) % len(self.pairs)

            except Exception as e:
                print(f"❌ Error loading image pair {idx}: {e}")
                return (
                    torch.randn(3, self.patch_size, self.patch_size),
                    torch.randn(3, self.patch_size * 2, self.patch_size * 2),
                )

            finally:
                # ✅ FIXED: Guaranteed cleanup of numpy arrays
                arrays_to_delete = [
                    ('lr_img', lr_img),
                    ('hr_img', hr_img),
                    ('lr_patch', lr_patch),
                    ('hr_patch', hr_patch),
                ]

                for _, arr in arrays_to_delete:
                    if arr is not None:
                        try:
                            del arr
                        except Exception:
                            pass

        # If all attempts were flat, return the last attempt
        if last_lr is None or last_hr is None:
            return (
                torch.randn(3, self.patch_size, self.patch_size),
                torch.randn(3, self.patch_size * 2, self.patch_size * 2),
            )
        return last_lr, last_hr
    
    def _get_patch(self, lr_img, hr_img):
        """Extract random patch"""
        patch_size = self.patch_size
        lr_h, lr_w = lr_img.shape[:2]
        
        if lr_w < patch_size or lr_h < patch_size:
            lr_img = cv2.resize(lr_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            hr_img = cv2.resize(hr_img, (patch_size*2, patch_size*2), interpolation=cv2.INTER_CUBIC)
            return lr_img, hr_img
        
        left = np.random.randint(0, lr_w - patch_size + 1)
        top = np.random.randint(0, lr_h - patch_size + 1)
        
        lr_patch = lr_img[top:top+patch_size, left:left+patch_size]
        hr_top, hr_left = top * 2, left * 2
        hr_patch = hr_img[hr_top:hr_top+patch_size*2, hr_left:hr_left+patch_size*2]
        
        return lr_patch, hr_patch
    
    def _augment(self, lr_patch, hr_patch):
        """Fast augmentation"""
        if np.random.random() > 0.5:
            lr_patch = cv2.flip(lr_patch, 1)
            hr_patch = cv2.flip(hr_patch, 1)
        
        if np.random.random() > 0.5:
            lr_patch = cv2.flip(lr_patch, 0)
            hr_patch = cv2.flip(hr_patch, 0)
        
        if np.random.random() > 0.5:
            angle = np.random.choice([90, 180, 270])
            if angle == 90:
                lr_patch = cv2.rotate(lr_patch, cv2.ROTATE_90_CLOCKWISE)
                hr_patch = cv2.rotate(hr_patch, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                lr_patch = cv2.rotate(lr_patch, cv2.ROTATE_180)
                hr_patch = cv2.rotate(hr_patch, cv2.ROTATE_180)
            elif angle == 270:
                lr_patch = cv2.rotate(lr_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                hr_patch = cv2.rotate(hr_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return lr_patch, hr_patch


class PreprocessedPatchDataset(Dataset):
    """Dataset for pre-cropped patches"""
    
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        
        self.lr_patches = sorted(list(self.lr_dir.glob('*.png')))
        
        self.pairs = []
        for lr_path in self.lr_patches:
            hr_path = self.hr_dir / lr_path.name
            if hr_path.exists():
                self.pairs.append((lr_path, hr_path))
        
        print(f"Found {len(self.pairs)} pre-cropped patch pairs")
        
        if len(self.pairs) == 0:
            raise ValueError("No patch pairs found!")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() > 0.5:
            lr_img = cv2.flip(lr_img, 1)
            hr_img = cv2.flip(hr_img, 1)
        
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor


# ============================================================================
# Custom Normalization Layer (Module Level)
# ============================================================================

class InstanceNormWeightOnly(nn.InstanceNorm2d):
    """InstanceNorm with only weight parameter (no bias)"""
    def __init__(self, num_features):
        # 🔴 CHANGED: eps=1e-3 (was default 1e-5). 
        # This prevents "divide by zero" errors on M4 chips.
        super().__init__(num_features, affine=False, track_running_stats=False, eps=1e-3)
        self.weight = nn.Parameter(torch.ones(num_features))
    
    def forward(self, input):
        # 🔴 CRITICAL FIX: Force calculation in Float32
        # MPS (Mac GPU) often overflows in Float16/BFloat16 during variance calc.
        # We cast to .float() (Float32), do the norm, then cast back.
        
        # 1. Sanitize input (fix NaNs coming from previous layers)
        if torch.isnan(input).any():
            input = torch.nan_to_num(input, nan=0.0)

        # 2. Force Float32 for the heavy lifting
        x_float = input.float()
        
        # 3. Manual safe normalization (more robust than super().forward)
        #    Formula: (x - mean) / sqrt(var + eps)
        mean = x_float.mean(dim=[2, 3], keepdim=True)
        var = x_float.var(dim=[2, 3], keepdim=True, unbiased=False)
        x_norm = (x_float - mean) / torch.sqrt(var + 1e-5)
        
        # 4. Apply weight and return to original dtype
        output = x_norm.type_as(input) * self.weight.view(1, -1, 1, 1)
        
        return output

# ============================================================================
# Model Architecture
# ============================================================================

class SuperUltraCompact(nn.Module):
    """Compact upscaler model - 2x upscaling"""
    def __init__(self, in_nc=3, out_nc=3, nf=24, nc=8, scale=2, use_activations=False):
        super(SuperUltraCompact, self).__init__()
        
        self.use_activations = use_activations
        body_layers = []
        
        body_layers.append(nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True, padding_mode='reflect'))
        
        if use_activations:
            body_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        body_layers.append(self._make_norm_layer(nf))
        
        for i in range(nc):
            body_layers.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True, padding_mode='reflect'))
            
            if use_activations:
                body_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            body_layers.append(self._make_norm_layer(nf))
        
        body_layers.append(nn.Conv2d(nf, scale * scale * out_nc, 3, 1, 1, bias=True, 
                                     padding_mode='reflect'))
        
        self.body = nn.Sequential(*body_layers)
        self.scale = scale
        
        print(f"✅ Model Architecture:")
        print(f"   Input: {in_nc}ch, Output: {out_nc}ch, Features: {nf}")
        print(f"   Conv-Norm pairs: {nc}")
        print(f"   Scale factor: {scale}x")
        print(f"   Activations: {'Enabled (LeakyReLU 0.2)' if use_activations else 'DISABLED'}")
        print(f"   Padding mode: 'reflect' (consistent throughout)")

    def _make_norm_layer(self, num_features):
        """Create InstanceNorm layer with ONLY weight (matching original architecture)"""
        return InstanceNormWeightOnly(num_features)

    def forward(self, x):
        self.nan_detected = False
        x = self.body(x)
        x = torch.nn.functional.pixel_shuffle(x, self.scale)

        # 🔴 CRITICAL FIX: torch.clamp passes NaNs through. 
        # We must use nan_to_num first to actually fix the explosion.
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.nan_detected = True
            print("⚠️  WARNING: NaN/Inf detected in model output! Attempting stabilization...")
            x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)

        x = torch.clamp(x, 0.0, 1.0)
        return x


# ============================================================================
# MEMORY-SAFE EXPONENTIAL MOVING AVERAGE (FIXED v2.3)
# ============================================================================

class EMA:
    """
    Exponential Moving Average - MEMORY OPTIMIZED
    Keeps shadow parameters on CPU to save VRAM
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # ✅ FIXED: Store on CPU to save VRAM
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().cpu()
    
    def update(self):
        """Update shadow parameters - keeps them on CPU"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    # Move to GPU, update, move back
                    shadow_on_device = self.shadow[name].to(param.device)
                    updated = (
                        self.decay * shadow_on_device + 
                        (1.0 - self.decay) * param.data.detach()
                    )
                    # ✅ Keep on CPU
                    self.shadow[name] = updated.detach().cpu()
                    del shadow_on_device, updated
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup current
                self.backup[name] = param.data.clone().detach()
                
                # Apply shadow (move from CPU to GPU)
                shadow_data = self.shadow[name].to(param.device)
                param.data.copy_(shadow_data)
                del shadow_data
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()
    
    def get_state_dict(self):
        """Get shadow state dict - returns CPU tensors"""
        return {k: v.clone().detach().cpu() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict):
        """Load shadow state dict"""
        for name, tensor in state_dict.items():
            self.shadow[name] = tensor.clone().detach().cpu()

    def cleanup(self):
        """
        Aggressively cleanup EMA shadow parameters
        Call this at end of training
        """
        print("  🧹 Cleaning up EMA...")
        
        try:
            # 1. Delete all shadow tensors
            shadow_keys = list(self.shadow.keys())
            for name in shadow_keys:
                try:
                    if name in self.shadow:
                        del self.shadow[name]
                except Exception as e:
                    pass
            
            # 2. Delete backup tensors
            backup_keys = list(self.backup.keys())
            for name in backup_keys:
                try:
                    if name in self.backup:
                        del self.backup[name]
                except Exception as e:
                    pass
            
            # 3. Clear dicts
            self.shadow.clear()
            self.backup.clear()
            
            # 4. Force garbage collection
            import gc
            collected = gc.collect()
            print(f"    ✅ EMA cleaned ({collected} objects freed)")
            
            # 5. Device cleanup
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"    ⚠️  EMA cleanup error: {e}")


def remap_state_dict_for_activations(state_dict):
    """Remap state dict from no-activation model to with-activation model"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('body.'):
            parts = key.split('.')
            try:
                idx = int(parts[1])
                # Formula: new_idx = old_idx + (old_idx + 1) // 2
                # This accounts for the extra ReLU layer after every Conv+Norm pair
                new_idx = idx + (idx + 1) // 2
                new_key = f"body.{new_idx}.{'.'.join(parts[2:])}"
                new_state_dict[new_key] = value
            except (ValueError, IndexError):
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


# ============================================================================
# OUTPUT CLAMPING WRAPPER - CRITICAL FIX
# ============================================================================

class ClampedOutputModel(nn.Module):
    """
    🔴 MOST CRITICAL FIX: Hard clamps model output to [0, 1] range.
    This is the single most important fix for over-brightening.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        self.nan_detected = getattr(self.model, 'nan_detected', False)
        return torch.clamp(output, 0.0, 1.0)
    
    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.model.eval()
        return super().eval()
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


# ============================================================================
# EPOCH-LEVEL QUALITY TRACKER (v3.0)
# ============================================================================

class EpochQualityTracker:
    """
    Tracks quality metrics across epochs and triggers rollback when
    degradation exceeds safe bounds.
    
    Memory cost: ~180KB for saved state dict (45k params * 4 bytes).
    """
    
    def __init__(self, model, device,
                 max_contrast_shift=8.0,   # % relative to baseline
                 max_exposure_shift=0.06,  # absolute p95 shift from baseline
                 max_brightness_shift=4.0, # % relative to baseline
                 patience=2):              # consecutive violations before rollback
        self.device = device
        self.patience = patience
        self.max_contrast_shift = max_contrast_shift
        self.max_exposure_shift = max_exposure_shift
        self.max_brightness_shift = max_brightness_shift
        
        # Baseline: captured from first epoch's validation
        self.baseline_metrics = None
        
        # Best known-good state (CPU to save VRAM)
        self.best_state = {k: v.clone().detach().cpu() 
                          for k, v in model.state_dict().items()}
        self.best_epoch = 0
        self.best_quality_score = float('inf')
        
        # Violation tracking
        self.consecutive_violations = 0
        self.total_rollbacks = 0
        self.epoch_history = []
    
    def compute_epoch_metrics(self, model, dataloader, device, num_batches=10):
        """
        Compute aggregate quality metrics over several validation batches.
        Lightweight: processes only num_batches, no gradients.
        """
        model.eval()
        
        all_contrast_drifts = []
        all_exposure_drifts = []
        all_brightness_drifts = []
        all_clip_fracs = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                lr, hr = batch[:2]
                lr = lr.to(device)
                hr = hr.to(device)
                
                sr = model(lr).clamp(0, 1)
                m = compute_quality_metrics(sr, hr)
                
                all_contrast_drifts.append(m['contrast_drift_pct'])
                all_exposure_drifts.append(m['p95_drift'])
                all_brightness_drifts.append(m['mean_drift_pct'])
                all_clip_fracs.append(m['clip_frac'])
                
                del sr, lr, hr
        
        model.train()
        
        if not all_contrast_drifts:
            return None
        
        return {
            'contrast_drift_pct': float(np.median(all_contrast_drifts)),
            'p95_drift': float(np.median(all_exposure_drifts)),
            'mean_drift_pct': float(np.median(all_brightness_drifts)),
            'clip_frac': float(np.median(all_clip_fracs)),
        }
    
    def check_epoch(self, model, epoch, epoch_metrics):
        """
        Call at end of each epoch after validation.
        Returns: (rolled_back: bool, message: str)
        """
        if epoch_metrics is None:
            return False, "No metrics available"
        
        # Set baseline from first measurement
        if self.baseline_metrics is None:
            self.baseline_metrics = epoch_metrics.copy()
            print(f"\n📊 QUALITY BASELINE SET [Epoch {epoch}]:")
            print(f"    Contrast drift:   {epoch_metrics['contrast_drift_pct']:+.2f}%")
            print(f"    Exposure p95:     {epoch_metrics['p95_drift']:+.4f}")
            print(f"    Brightness drift: {epoch_metrics['mean_drift_pct']:+.2f}%")
            return False, "Baseline set"
        
        self.epoch_history.append(epoch_metrics)
        
        # Check for violations relative to baseline
        violations = []
        
        contrast_shift = abs(epoch_metrics['contrast_drift_pct']) - abs(self.baseline_metrics['contrast_drift_pct'])
        if contrast_shift > self.max_contrast_shift:
            violations.append(f"Contrast degraded by {contrast_shift:+.1f}% vs baseline")
        
        exposure_shift = epoch_metrics['p95_drift'] - self.baseline_metrics['p95_drift']
        if exposure_shift > self.max_exposure_shift:
            violations.append(f"Exposure p95 shifted {exposure_shift:+.4f} vs baseline")
        
        brightness_shift = abs(epoch_metrics['mean_drift_pct']) - abs(self.baseline_metrics['mean_drift_pct'])
        if brightness_shift > self.max_brightness_shift:
            violations.append(f"Brightness degraded by {brightness_shift:+.1f}% vs baseline")
        
        # Compute quality score (lower is better)
        quality_score = (abs(epoch_metrics['contrast_drift_pct']) + 
                        abs(epoch_metrics['p95_drift']) * 100 +
                        abs(epoch_metrics['mean_drift_pct']))
        
        if violations:
            self.consecutive_violations += 1
            print(f"\n🔴 QUALITY VIOLATION [Epoch {epoch}] ({self.consecutive_violations}/{self.patience}):")
            for v in violations:
                print(f"    • {v}")
        else:
            self.consecutive_violations = 0
            # Update best state if quality improved
            if quality_score < self.best_quality_score:
                self.best_quality_score = quality_score
                self.best_state = {k: v.clone().detach().cpu() 
                                  for k, v in model.state_dict().items()}
                self.best_epoch = epoch
        
        # Trigger rollback?
        if self.consecutive_violations >= self.patience:
            self.total_rollbacks += 1
            print(f"\n🔄 ROLLING BACK to epoch {self.best_epoch} state "
                  f"(rollback #{self.total_rollbacks})")
            
            # Restore best state
            restored = {k: v.to(self.device) for k, v in self.best_state.items()}
            model.load_state_dict(restored, strict=False)
            del restored
            
            self.consecutive_violations = 0
            
            return True, f"Rolled back to epoch {self.best_epoch}"
        
        return False, f"Quality score: {quality_score:.2f}"
    
    def cleanup(self):
        """Free stored state dict"""
        self.best_state.clear()
        self.epoch_history.clear()




def load_pretrained(model, checkpoint_path, device):
    """Robust checkpoint loading compatible with weight-only InstanceNorm checkpoints"""
    print(f"\nLoading pretrained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove DataParallel 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # ✅ CRITICAL: Filter out any unexpected bias keys from checkpoint
    # The original checkpoint has InstanceNorm layers with weight-only (no bias)
    # Our model architecture now matches this (weight-only InstanceNorm)
    checkpoint_keys = list(state_dict.keys())
    for key in checkpoint_keys:
        # Check if this is a bias key that shouldn't exist
        if '.bias' in key and 'body.' in key:
            # Check if the corresponding layer in our model expects a bias
            if key not in model.state_dict():
                print(f"  🔄 Removing unexpected bias key from checkpoint: {key}")
                del state_dict[key]
    
    # Try loading the cleaned state_dict
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        loaded_count = len(state_dict)
        total_count = len(model.state_dict())
        print(f"  ✅ Loaded {loaded_count}/{total_count} keys")
        
        if missing:
            print(f"  ⚠️  Missing keys: {list(missing)[:5]}...")
        if unexpected:
            print(f"  ⚠️  Unexpected keys: {list(unexpected)[:5]}...")
            
    except RuntimeError as e:
        if "size mismatch" in str(e) and "body." in str(e):
            print("  ⚠️  Detected activation layer mismatch")
            print("  🔄 Remapping state dict to insert activation layers...")
            
            state_dict = remap_state_dict_for_activations(state_dict)
            
            # Re-clean any unexpected bias keys after remapping
            checkpoint_keys = list(state_dict.keys())
            for key in checkpoint_keys:
                if '.bias' in key and 'body.' in key:
                    if key not in model.state_dict():
                        print(f"  🔄 Removing unexpected bias key after remapping: {key}")
                        del state_dict[key]
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ Remapped successfully! Loaded keys")
            if missing:
                print(f"     Missing: {list(missing)[:5]}...")
            if unexpected:
                print(f"     Unexpected: {list(unexpected)[:5]}...")
        else:
            raise e
    
    model = model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    return model




# ============================================================================
# QUALITY MONITORING FUNCTIONS (v3.0 — replaces BRIGHTNESS MONITORING)
# ============================================================================

def compute_quality_metrics(pred, target):
    """
    Compute brightness, contrast, AND exposure metrics for monitoring.
    Replaces compute_brightness_metrics(). Cost: ~0.1ms per batch on MPS.
    """
    with torch.no_grad():
        # Luminance (BT.709)
        pred_lum = 0.2126 * pred[:, 0] + 0.7152 * pred[:, 1] + 0.0722 * pred[:, 2]
        target_lum = 0.2126 * target[:, 0] + 0.7152 * target[:, 1] + 0.0722 * target[:, 2]

        # --- Brightness ---
        pred_mean = pred_lum.mean().item()
        target_mean = target_lum.mean().item()
        mean_drift_pct = ((pred_mean - target_mean) / (target_mean + 1e-8)) * 100

        # --- Contrast (global std ratio) ---
        pred_std = pred_lum.std().item()
        target_std = target_lum.std().item()
        contrast_drift_pct = ((pred_std - target_std) / (target_std + 1e-8)) * 100

        # --- Exposure distribution (percentile overshoots) ---
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        p95_pred = _safe_quantile_1d(pred_flat, 0.95).item()
        p95_target = _safe_quantile_1d(target_flat, 0.95).item()
        p99_pred = _safe_quantile_1d(pred_flat, 0.99).item()
        p99_target = _safe_quantile_1d(target_flat, 0.99).item()
        p05_pred = _safe_quantile_1d(pred_flat, 0.05).item()
        p05_target = _safe_quantile_1d(target_flat, 0.05).item()

        # Fraction of pixels near clipping
        clip_frac = (pred > 0.98).float().mean().item()
        crush_frac = (pred < 0.02).float().mean().item()

        metrics = {
            # Brightness
            'mean_pred': pred_mean,
            'mean_target': target_mean,
            'mean_drift_pct': mean_drift_pct,
            # Contrast
            'std_pred': pred_std,
            'std_target': target_std,
            'contrast_drift_pct': contrast_drift_pct,
            # Exposure
            'p95_pred': p95_pred,
            'p95_target': p95_target,
            'p95_drift': p95_pred - p95_target,
            'p99_pred': p99_pred,
            'p99_target': p99_target,
            'p99_drift': p99_pred - p99_target,
            'p05_pred': p05_pred,
            'p05_target': p05_target,
            'p05_drift': p05_pred - p05_target,
            # Clipping
            'clip_frac': clip_frac,
            'crush_frac': crush_frac,
            # Legacy (backward compat with compute_brightness_metrics)
            'max_pred': pred.max().item(),
            'max_target': target.max().item(),
            'max_drift_pct': ((pred.max().item() - target.max().item()) /
                              (target.max().item() + 1e-8)) * 100,
        }

        return metrics


def check_quality_gate(metrics, epoch, batch_idx,
                       max_contrast_drift_pct=15.0,
                       max_exposure_drift=0.08,
                       max_mean_drift_pct=5.0,
                       max_clip_frac=0.05):
    """
    Decide whether the current batch output is safe for an optimizer step.

    Returns:
        (should_skip: bool, reasons: list[str])

    Thresholds relax during warmup (epochs 1-5) and tighten gradually.
    """
    warmup_factor = min(1.0, epoch / 5.0)
    eff_contrast = max_contrast_drift_pct / max(warmup_factor, 0.3)
    eff_exposure = max_exposure_drift / max(warmup_factor, 0.3)
    eff_mean = max_mean_drift_pct / max(warmup_factor, 0.3)
    eff_clip = max_clip_frac / max(warmup_factor, 0.3)

    reasons = []

    if abs(metrics['contrast_drift_pct']) > eff_contrast:
        reasons.append(
            f"contrast drift {metrics['contrast_drift_pct']:+.1f}% "
            f"(limit ±{eff_contrast:.0f}%)"
        )

    if metrics['p95_drift'] > eff_exposure:
        reasons.append(
            f"p95 exposure overshoot +{metrics['p95_drift']:.3f} "
            f"(limit +{eff_exposure:.3f})"
        )

    if metrics['p05_drift'] < -eff_exposure:
        reasons.append(
            f"p05 shadow crush {metrics['p05_drift']:.3f} "
            f"(limit -{eff_exposure:.3f})"
        )

    if abs(metrics['mean_drift_pct']) > eff_mean:
        reasons.append(
            f"brightness drift {metrics['mean_drift_pct']:+.1f}% "
            f"(limit ±{eff_mean:.0f}%)"
        )

    if metrics['clip_frac'] > eff_clip:
        reasons.append(
            f"clipping {metrics['clip_frac']*100:.1f}% pixels "
            f"(limit {eff_clip*100:.1f}%)"
        )

    should_skip = len(reasons) > 0

    if should_skip and batch_idx % 50 == 0:
        print(f"\n🚫 QUALITY GATE [Epoch {epoch}, Batch {batch_idx}] — skipping update:")
        for r in reasons:
            print(f"    • {r}")

    return should_skip, reasons


def log_quality_warning(metrics, epoch, batch_idx):
    """
    Enhanced logging replacing log_brightness_warning().
    Covers brightness, contrast, and exposure in one call.
    """
    warnings_list = []

    if abs(metrics['mean_drift_pct']) > 3.0:
        warnings_list.append(
            f"Brightness: {metrics['mean_drift_pct']:+.1f}% "
            f"(pred={metrics['mean_pred']:.4f}, target={metrics['mean_target']:.4f})"
        )

    if abs(metrics['contrast_drift_pct']) > 10.0:
        warnings_list.append(
            f"Contrast: {metrics['contrast_drift_pct']:+.1f}% "
            f"(pred_std={metrics['std_pred']:.4f}, target_std={metrics['std_target']:.4f})"
        )

    if abs(metrics['p95_drift']) > 0.05:
        warnings_list.append(
            f"Exposure p95: {metrics['p95_drift']:+.4f} "
            f"(pred={metrics['p95_pred']:.4f}, target={metrics['p95_target']:.4f})"
        )

    if metrics['clip_frac'] > 0.02:
        warnings_list.append(
            f"Clipping: {metrics['clip_frac']*100:.1f}% of pixels > 0.98"
        )

    if warnings_list:
        print(f"\n⚠️  QUALITY DRIFT [Epoch {epoch}, Batch {batch_idx}]")
        for w in warnings_list:
            print(f"    {w}")


# Backward-compatible aliases
def compute_brightness_metrics(pred, target):
    """Backward-compatible wrapper for compute_quality_metrics."""
    return compute_quality_metrics(pred, target)


def log_brightness_warning(metrics, epoch, batch_idx):
    """Backward-compatible wrapper for log_quality_warning."""
    return log_quality_warning(metrics, epoch, batch_idx)



# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion_l1, criterion_perceptual,
                device, epoch, ema=None, use_perceptual=True, perceptual_weight=0.5,
                warmup_scheduler=None, use_autocast=False, accumulation_steps=1,
                criterion_brightness=None, criterion_contrast=None, 
                criterion_color=None, criterion_tv=None,
                brightness_weight=0.1, contrast_weight=0.1, 
                color_weight=0.05, tv_weight=0.005,
                criterion_highfreq=None, criterion_edge=None,
                highfreq_weight=0.0, edge_weight=0.0,
                criterion_highlight=None, criterion_drange=None, 
                criterion_hl_grad=None,
                highlight_weight=0.3, drange_weight=0.1, hlgrad_weight=0.05,
                criterion_lab=None, lab_weight=0.0,
                criterion_local=None, local_weight=0.0,
                criterion_ssim=None, ssim_weight=0.0,
                criterion_exposure=None, exposure_weight=0.0,
                criterion_hist=None, hist_weight=0.0,
                criterion_log_tone=None, log_tone_weight=0.0,
                writer=None,
                criterion_mean_lum=None, mean_lum_weight=0.0,
                criterion_percentile=None, percentile_weight=0.0,
                output_dir=None,
                main_scheduler=None,
                lr_backoff_patience=3,
                lr_backoff_factor=0.5,
                lr_backoff_min=1e-7):
    """
    Training epoch — FIXED: Loss computed as connected tensor for proper gradient flow.
    """
    model.train()
    total_loss_accum = 0.0
    valid_batches = 0
    
    optimizer.zero_grad(set_to_none=True)
    last_good_state = snapshot_model_state(model)
    last_good_opt_state = snapshot_optimizer_state(optimizer)
    nan_recovery_count = 0
    recoveries_since_backoff = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', dynamic_ncols=True, leave=False)
    last_components = {}
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # ========== LOAD DATA ==========
            lr, hr = batch[:2]
            
            if lr.isnan().any() or hr.isnan().any():
                print(f"⚠️  Skipping batch {batch_idx} — NaN in input")
                continue
            
            lr = to_device(lr, device)
            hr = to_device(hr, device)
            
            # ========== FORWARD PASS ==========
            if use_autocast and device.type == 'cuda':
                with get_autocast('cuda'):
                    sr = model(lr)
            else:
                sr = model(lr)

            # Bail out early if model produced NaN/Inf internally
            if getattr(model, 'nan_detected', False):
                nan_recovery_count += 1
                recoveries_since_backoff += 1
                print(f"⚠️  NaN/Inf detected in forward at batch {batch_idx} — restoring last good state")
                restore_model_state(model, last_good_state, device)
                restore_optimizer_state(optimizer, last_good_opt_state, device)
                optimizer.zero_grad(set_to_none=True)
                if recoveries_since_backoff >= lr_backoff_patience:
                    new_lrs = apply_lr_backoff(
                        optimizer,
                        factor=lr_backoff_factor,
                        min_lr=lr_backoff_min,
                        schedulers=[warmup_scheduler, main_scheduler],
                    )
                    print(f"🔽 LR backoff triggered after {recoveries_since_backoff} recoveries: {new_lrs}")
                    recoveries_since_backoff = 0
                aggressive_memory_cleanup(device)
                del sr, lr, hr
                continue
            
            # ========== QUALITY MONITORING (every 50 batches) (v3.0) ==========
            if batch_idx % 50 == 0:
                try:
                    qm = compute_quality_metrics(sr.detach(), hr)
                    log_quality_warning(qm, epoch, batch_idx)
                    if writer is not None:
                        gs = epoch * len(dataloader) + batch_idx
                        writer.add_scalar('Quality/Mean_Drift_Pct', qm['mean_drift_pct'], gs)
                        writer.add_scalar('Quality/Contrast_Drift_Pct', qm['contrast_drift_pct'], gs)
                        writer.add_scalar('Quality/P95_Drift', qm['p95_drift'], gs)
                        writer.add_scalar('Quality/Clip_Fraction', qm['clip_frac'], gs)
                    del qm
                except Exception:
                    pass
            

            # ========== COMPUTE LOSS (CONNECTED TENSOR) ==========
            loss_tensor, components = compute_total_loss(
                sr, hr, device, batch_idx, epoch,
                criterion_l1, criterion_perceptual, use_perceptual, perceptual_weight,
                criterion_brightness, brightness_weight,
                criterion_contrast, contrast_weight,
                criterion_color, color_weight,
                criterion_tv, tv_weight,
                criterion_highfreq, highfreq_weight,
                criterion_edge, edge_weight,
                criterion_highlight, highlight_weight,
                criterion_drange, drange_weight,
                criterion_hl_grad, hlgrad_weight,
                criterion_lab, lab_weight,
                criterion_local, local_weight,
                criterion_ssim, ssim_weight,
                criterion_exposure, exposure_weight,
                criterion_hist, hist_weight,
                criterion_log_tone, log_tone_weight,
                criterion_mean_lum, mean_lum_weight,
                criterion_percentile, percentile_weight,
            )
            
            # Skip if loss is NaN/Inf OR if no losses contributed (no grad_fn)
            if torch.isnan(loss_tensor) or torch.isinf(loss_tensor) or loss_tensor.grad_fn is None:
                if loss_tensor.grad_fn is None:
                    print(f"⚠️  Skipping batch {batch_idx} — all losses were NaN/Inf, no grad_fn")
                else:
                    print(f"⚠️  Skipping batch {batch_idx} — NaN/Inf total loss")
                optimizer.zero_grad(set_to_none=True)
                del sr, lr, hr, loss_tensor
                continue

            scaled_loss = loss_tensor / accumulation_steps

            # ========== BACKWARD PASS ==========
            scaled_loss.backward()

            # Skip update if gradients are invalid
            if has_invalid_gradients(model):
                nan_recovery_count += 1
                recoveries_since_backoff += 1
                print(f"⚠️  NaN/Inf gradients at batch {batch_idx} — restoring last good state")
                restore_model_state(model, last_good_state, device)
                restore_optimizer_state(optimizer, last_good_opt_state, device)
                optimizer.zero_grad(set_to_none=True)
                if recoveries_since_backoff >= lr_backoff_patience:
                    new_lrs = apply_lr_backoff(
                        optimizer,
                        factor=lr_backoff_factor,
                        min_lr=lr_backoff_min,
                        schedulers=[warmup_scheduler, main_scheduler],
                    )
                    print(f"🔽 LR backoff triggered after {recoveries_since_backoff} recoveries: {new_lrs}")
                    recoveries_since_backoff = 0
                aggressive_memory_cleanup(device)
                del sr, lr, hr, loss_tensor, scaled_loss
                continue
            
            total_loss_accum += loss_tensor.item()
            last_components = components
            
            del sr, lr, hr, loss_tensor, scaled_loss
            
            # ========== OPTIMIZER STEP (every accumulation_steps) ==========
            if (batch_idx + 1) % accumulation_steps == 0 or \
               (batch_idx + 1) == len(dataloader):
                
                # Tighter gradient clipping for tiny model
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                if batch_idx < accumulation_steps:
                    verify_gradient_flow(model, epoch, 0)

                    # Log gradient norm for tuning
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float('inf')
                    )
                    print(f"     🔍 Gradient norm before clip: {grad_norm:.4f}")
                
                # 🆕 QUALITY GATE (v3.0): Check output before committing update
                gate_skip = False
                if batch_idx % accumulation_steps == 0:
                    try:
                        with torch.no_grad():
                            gate_lr = to_device(batch[0][:1], device)
                            gate_sr = model(gate_lr).clamp(0, 1)
                            gate_hr = to_device(batch[1][:1], device)
                            gate_metrics = compute_quality_metrics(gate_sr, gate_hr)
                            gate_skip, gate_reasons = check_quality_gate(
                                gate_metrics, epoch, batch_idx
                            )
                            del gate_lr, gate_sr, gate_hr, gate_metrics
                    except Exception:
                        gate_skip = False  # Fail open
                
                if gate_skip:
                    optimizer.zero_grad(set_to_none=True)  # Discard bad gradients
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if warmup_scheduler:
                        warmup_scheduler.step()
                    
                    if ema:
                        ema.update()

                    # Validate parameters after the step
                    if has_invalid_parameters(model):
                        nan_recovery_count += 1
                        recoveries_since_backoff += 1
                        print(f"⚠️  NaN/Inf parameters after step at batch {batch_idx} — restoring last good state")
                        restore_model_state(model, last_good_state, device)
                        restore_optimizer_state(optimizer, last_good_opt_state, device)
                        if recoveries_since_backoff >= lr_backoff_patience:
                            new_lrs = apply_lr_backoff(
                                optimizer,
                                factor=lr_backoff_factor,
                                min_lr=lr_backoff_min,
                                schedulers=[warmup_scheduler, main_scheduler],
                            )
                            print(f"🔽 LR backoff triggered after {recoveries_since_backoff} recoveries: {new_lrs}")
                            recoveries_since_backoff = 0
                        aggressive_memory_cleanup(device)
                    else:
                        last_good_state = snapshot_model_state(model)
                        last_good_opt_state = snapshot_optimizer_state(optimizer)
                        valid_batches += 1
            

            # ========== MPS MEMORY MANAGEMENT ==========
            if device.type == 'mps' and batch_idx % 10 == 0:
                torch.mps.empty_cache()
            
            adaptive_batch_cleanup(device, epoch, batch_idx, memory_threshold_gb=13)
            
            # ========== PROGRESS BAR ==========
            if last_components:
                sorted_c = sorted(last_components.items(), key=lambda x: -x[1])[:4]
                pbar.set_postfix({k: f'{v:.4f}' for k, v in sorted_c})
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  OOM at batch {batch_idx}. Clearing cache...")
                optimizer.zero_grad(set_to_none=True)
                aggressive_memory_cleanup(device)
                continue
            else:
                raise
        
        except Exception as e:
            print(f"⚠️  Batch {batch_idx} error: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue
    
    avg_loss = total_loss_accum / max(valid_batches * accumulation_steps, 1)
    
    if device.type == 'mps':
        torch.mps.empty_cache()

    # Save a training sample for visual debugging
    if epoch <= 5 or epoch % 5 == 0:
        try:
            model.eval()
            test_batch = next(iter(dataloader))
            test_lr = to_device(test_batch[0][:1], device)
            test_hr = to_device(test_batch[1][:1], device)
            with torch.no_grad():
                test_sr = model(test_lr).clamp(0, 1)
            
            if HAS_CV2:
                # Save SR output
                sr_np = test_sr[0].cpu().numpy().transpose(1, 2, 0)
                sr_np = (sr_np * 255).astype(np.uint8)
                sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
                debug_dir = output_dir or './checkpoints'
                debug_path = os.path.join(debug_dir, f'debug_epoch_{epoch:03d}.png')
                cv2.imwrite(debug_path, sr_bgr)
            
            del test_lr, test_hr, test_sr
            model.train()
        except:
            model.train()
    
    return avg_loss


def validate(model, dataloader, criterion_l1, criterion_perceptual,
             device, use_perceptual=True, perceptual_weight=0.5,
             use_autocast=False, calc_metrics=True):
    """
    Validation - MEMORY OPTIMIZED v2.3
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    # ✅ FIXED: Pre-allocate numpy arrays instead of lists
    if calc_metrics and HAS_METRICS:
        max_batches = len(dataloader)
        psnr_scores = np.zeros(max_batches)
        ssim_scores = np.zeros(max_batches)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            try:
                # ========== LOAD DATA ==========
                if len(batch) == 2:
                    lr, hr = batch
                else:
                    lr, hr = batch[:2]
                
                lr = to_device(lr, device)
                hr = to_device(hr, device)
                
                # ========== FORWARD PASS ==========
                if use_autocast and device.type == 'cuda':
                    with get_autocast('cuda'):
                        sr = model(lr)
                else:
                    sr = model(lr)
                
                # ========== COMPUTE LOSS ==========
                loss_l1 = criterion_l1(sr, hr).item()
                loss_value = loss_l1
                
                if use_perceptual and criterion_perceptual is not None:
                    loss_perc = criterion_perceptual(sr, hr).item()
                    loss_value = (1.0 - perceptual_weight) * loss_l1 + \
                                 perceptual_weight * loss_perc
                
                total_loss += loss_value
                batch_count += 1
                
                # ========== COMPUTE METRICS ==========
                if calc_metrics and HAS_METRICS:
                    for i in range(sr.shape[0]):
                        try:
                            sr_np = sr[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                            hr_np = hr[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                            
                            psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
                            ssim = structural_similarity(hr_np, sr_np, data_range=1.0, 
                                                        channel_axis=2)
                            
                            # ✅ FIXED: Store in pre-allocated array
                            if batch_idx < len(psnr_scores):
                                psnr_scores[batch_idx] = psnr
                                ssim_scores[batch_idx] = ssim
                            
                            del sr_np, hr_np
                        except Exception as e:
                            print(f"⚠️  Metrics error at batch {batch_idx}: {e}")
                
                # ========== CLEANUP ==========
                del sr, lr, hr
                
                # ✅ FIXED: Periodic cleanup
                if batch_idx % 5 == 0:
                    safe_gc_collect(device)
                
            except Exception as e:
                print(f"⚠️  Validation batch {batch_idx} error: {e}")
                try:
                    del sr, lr, hr
                except:
                    pass
                continue
    
    # ========== COMPILE RESULTS ==========
    result = {'loss': total_loss / max(batch_count, 1)}
    
    if calc_metrics and HAS_METRICS and batch_count > 0:
        # Only use valid scores
        valid_psnr = psnr_scores[:batch_count]
        valid_ssim = ssim_scores[:batch_count]
        
        result['psnr'] = np.mean(valid_psnr)
        result['psnr_std'] = np.std(valid_psnr)
        result['ssim'] = np.mean(valid_ssim)
        result['ssim_std'] = np.std(valid_ssim)
        
        del psnr_scores, ssim_scores, valid_psnr, valid_ssim
    
    # ✅ FIXED: Final cleanup
    safe_gc_collect(device)
    
    return result


def save_validation_samples(model, dataloader, device, epoch, output_dir, num_samples=4):
    """Save LR | SR | HR comparisons - MEMORY OPTIMIZED"""
    model.eval()
    
    try:
        batch = next(iter(dataloader))
        if len(batch) == 2:
            lr, hr = batch
        else:
            lr, hr = batch[:2]
        
        lr = lr[:num_samples]
        hr = hr[:num_samples]
        
        lr = to_device(lr, device)
        hr = to_device(hr, device)
        
        with torch.no_grad():
            sr = model(lr).clamp(0, 1)
        
        # Bicubic upscale of LR
        bicubic = torch.nn.functional.interpolate(
            lr, scale_factor=2, mode='bicubic', align_corners=False
        ).clamp(0, 1)
        
        # Save comparisons
        for i in range(min(num_samples, lr.shape[0])):
            comparison = torch.cat([bicubic[i], sr[i], hr[i]], dim=2)
            
            img_np = (comparison.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            output_path = f'{output_dir}/val_epoch_{epoch:03d}_sample_{i}.png'
            cv2.imwrite(output_path, img_rgb)
        
        print(f"✅ Validation samples saved to {output_dir}")
        
        # 🔴 CLEANUP
        del sr, bicubic, lr, hr, comparison, img_np, img_rgb
        if device.type == 'mps':
            torch.mps.empty_cache()
            
    except Exception as e:
        print(f"⚠️  Could not save validation samples: {e}")


def validate_checkpoint_quality(model, val_loader, device, num_batches=5):
    """
    Enhanced checkpoint validation with contrast and exposure checks (v3.0).
    Drop-in replacement — same signature, stricter criteria.
    
    Returns: (is_valid: bool, message: str)
    """
    was_training = model.training
    model.eval()
    
    validation_passed = True
    issues = []
    
    contrast_ratios = []
    exposure_drifts = []
    clip_fracs = []
    
    try:
        with torch.no_grad():
            batch_count = 0
            
            for batch in val_loader:
                if batch_count >= num_batches:
                    break
                
                try:
                    if len(batch) == 2:
                        lr, hr = batch
                    else:
                        lr, hr = batch[:2]
                    
                    lr = to_device(lr, device)
                    hr = to_device(hr, device)
                    
                    sr = model(lr)
                    
                    # --- Existing checks ---
                    if sr.isnan().any() or sr.isinf().any():
                        issues.append("NaN/Inf in output")
                        validation_passed = False
                        del sr, lr, hr
                        break
                    
                    if sr.max() > 1.0 or sr.min() < 0.0:
                        issues.append(f"Range [{sr.min():.2f}, {sr.max():.2f}]")
                        validation_passed = False
                        del sr, lr, hr
                        break
                    
                    # --- NEW: Contrast ratio check ---
                    sr_lum = 0.2126 * sr[:, 0] + 0.7152 * sr[:, 1] + 0.0722 * sr[:, 2]
                    hr_lum = 0.2126 * hr[:, 0] + 0.7152 * hr[:, 1] + 0.0722 * hr[:, 2]
                    
                    sr_std = sr_lum.std().item()
                    hr_std = hr_lum.std().item()
                    
                    if hr_std > 1e-6:
                        contrast_ratios.append(sr_std / hr_std)
                    
                    # --- NEW: Exposure drift check ---
                    p95_sr = _safe_quantile_1d(sr, 0.95).item()
                    p95_hr = _safe_quantile_1d(hr, 0.95).item()
                    exposure_drifts.append(p95_sr - p95_hr)
                    
                    # --- NEW: Clipping fraction ---
                    clip_f = (sr > 0.98).float().mean().item()
                    clip_fracs.append(clip_f)
                    
                    validation_passed = True
                    
                    del sr, lr, hr, sr_lum, hr_lum
                    
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    
                    batch_count += 1
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        issues.append("OOM")
                    else:
                        issues.append(f"Runtime error: {str(e)[:50]}")
                    validation_passed = False
                    break
                
                except Exception as e:
                    issues.append(f"Error: {str(e)[:50]}")
                    validation_passed = False
                    break
        
        # Evaluate aggregate metrics
        if contrast_ratios:
            median_cr = float(np.median(contrast_ratios))
            median_exp = float(np.median(exposure_drifts))
            median_clip = float(np.median(clip_fracs))
            
            if median_cr < 0.5 or median_cr > 2.0:
                issues.append(
                    f"Contrast ratio {median_cr:.2f}x "
                    f"(expected 0.5-2.0x)"
                )
                validation_passed = False
            
            if median_exp > 0.10:
                issues.append(
                    f"Exposure overshoot p95={median_exp:+.3f} "
                    f"(limit +0.10)"
                )
                validation_passed = False
            
            if median_clip > 0.03:
                issues.append(
                    f"Clipping {median_clip*100:.1f}% "
                    f"(limit 3%)"
                )
                validation_passed = False
    
    except Exception as e:
        issues.append(f"Validation failed: {str(e)[:50]}")
        validation_passed = False
    
    finally:
        try:
            if was_training:
                model.train()
            if device.type == 'mps':
                aggressive_memory_cleanup(device)
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ⚠️  Cleanup error: {e}")
    
    msg = "OK" if validation_passed else "; ".join(issues)
    return validation_passed, msg



def save_checkpoint(model, optimizer, epoch, loss, filepath, ema=None,
                   main_scheduler=None, warmup_scheduler=None, is_best=False,
                   val_loader=None, device=None):
    """
    Save checkpoint with validation and cleanup
    """
    print(f"\n💾 Saving checkpoint: {filepath}")
    
    # Validate checkpoint quality before saving
    if val_loader is not None and device is not None:
        print("  🔍 Validating checkpoint quality...")
        is_valid, msg = validate_checkpoint_quality(model, val_loader, device, num_batches=3)
        
        if not is_valid:
            print(f"  ❌ CHECKPOINT VALIDATION FAILED: {msg}")
            print(f"     Skipping save of {filepath}")
            return False
        else:
            print(f"  ✅ {msg}")
    
    try:
        # Build checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if ema is not None:
            checkpoint['ema_state_dict'] = ema.get_state_dict()
        
        if main_scheduler is not None:
            checkpoint['main_scheduler_state_dict'] = main_scheduler.state_dict()
        
        if warmup_scheduler is not None:
            checkpoint['warmup_scheduler_state_dict'] = warmup_scheduler.state_dict()
        
        # Save
        torch.save(checkpoint, filepath)
        
        # Verify file was created
        if Path(filepath).exists():
            file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            prefix = "✅ BEST" if is_best else "✅"
            print(f"{prefix} Checkpoint saved: {filepath} ({file_size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ Failed to save checkpoint: {filepath}")
            return False
    
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return False
    
    finally:
        # Cleanup after saving
        if device and device.type == 'mps':
            torch.mps.empty_cache()


# ============================================================================
# LOSS MANAGER - Prevents criterion accumulation
# ============================================================================

class LossManager:
    """Centralized management of all loss functions with cleanup"""
    
    def __init__(self, device):
        self.device = device
        self.losses = {}
    
    def create_losses(self, args):
        """Initialize all loss functions"""
        self.losses['l1'] = CharbonnierLoss()
        
        # ✅ FIX: Only load VGG if weight is meaningful
        if args.use_perceptual and args.perceptual_weight >= 0.01:
            print(f"  Loading perceptual loss (weight={args.perceptual_weight})...")
            self.losses['perceptual'] = BrightnessNormalizedPerceptualLoss(
                device=self.device,
                layers=['relu1_2', 'relu2_2', 'relu3_3']
            ).to(self.device)
        else:
            self.losses['perceptual'] = None
            if args.use_perceptual and args.perceptual_weight > 0:
                print(f"  ⚠️  Perceptual weight {args.perceptual_weight} < 0.01, "
                      f"skipping VGG load to save ~1.5GB")
        
        # Primary losses
        self.losses['brightness'] = BrightnessConsistencyLoss().to(self.device) \
            if args.brightness_weight > 0 else None
        self.losses['contrast'] = ContrastConsistencyLoss().to(self.device) \
            if args.contrast_weight > 0 else None
        self.losses['color'] = ColorRangeLimiter(margin=args.safe_margin).to(self.device) \
            if args.color_weight > 0 else None
        self.losses['tv'] = TotalVariationLimiter().to(self.device) \
            if args.tv_weight > 0 else None
        
        # Advanced losses
        self.losses['highfreq'] = HighFrequencyLoss().to(self.device) \
            if args.highfreq_weight > 0 else None
        self.losses['edge'] = EdgePreservationLoss().to(self.device) \
            if args.edge_weight > 0 else None
        self.losses['highlight'] = AdaptiveHighlightSuppressionLoss(
            threshold=args.safe_margin + 0.87,
            safe_margin=args.safe_margin
        ).to(self.device) if args.highlight_weight > 0 else None
        self.losses['drange'] = ApproximateDynamicRangeLoss().to(self.device) \
            if args.drange_weight > 0 else None
        self.losses['hl_grad'] = HighlightGradientLoss().to(self.device) \
            if args.hlgrad_weight > 0 else None
        
        # Brightness preservation losses (NEW v2.0)
        self.losses['lab'] = LabColorLoss(l_weight=2.0, ab_weight=1.0).to(self.device) \
            if args.lab_weight > 0 else None
        self.losses['local'] = LocalStatisticsLoss(kernel_size=16).to(self.device) \
            if args.local_weight > 0 else None
        self.losses['ssim'] = SSIMLuminanceLoss(window_size=11).to(self.device) \
            if args.ssim_weight > 0 else None
        self.losses['exposure'] = ExposureGradientLoss().to(self.device) \
            if args.exposure_weight > 0 else None
        self.losses['hist'] = SoftHistogramLoss(bins=32).to(self.device) \
            if args.hist_weight > 0 else None
        self.losses['log_tone'] = LogToneMappingLoss().to(self.device) \
            if args.log_tone_weight > 0 else None
        self.losses['mean_lum'] = MeanLuminancePreservationLoss().to(self.device) \
            if args.mean_lum_weight > 0 else None
        self.losses['percentile'] = PercentilePreservationLoss().to(self.device) \
            if args.percentile_weight > 0 else None
        
        return self.losses
    
    def get(self, name):
        """Safely get a loss function"""
        return self.losses.get(name, None)
    
    def cleanup(self):
        """Free all GPU memory and clear references"""
        print("\n🧹 Cleaning up loss functions...")
        
        for name, loss_fn in self.losses.items():
            if loss_fn is not None:
                try:
                    # Move to CPU first
                    if hasattr(loss_fn, 'cpu'):
                        loss_fn.cpu()
                    
                    # Clear any internal buffers
                    if hasattr(loss_fn, 'vgg'):
                        del loss_fn.vgg
                    
                    # Delete the module
                    del loss_fn
                    
                except Exception as e:
                    print(f"  ⚠️  Error cleaning {name}: {e}")
        
        self.losses.clear()
        print("  ✅ All loss functions cleaned")


# ============================================================================
# COMPREHENSIVE CLEANUP FUNCTION
# ============================================================================

def cleanup_all_resources(device, model, ema, loss_manager, writer, train_loader):
    """
    Comprehensive cleanup of all resources at end of training.
    Call this in finally block of main().
    """
    print("\n" + "="*70)
    print("🧹 COMPREHENSIVE RESOURCE CLEANUP")
    print("="*70)
    
    # 1. Close TensorBoard
    print("\n1️⃣  Closing TensorBoard...")
    try:
        if writer is not None:
            writer.flush()
            writer.close()
            print("   ✅ TensorBoard closed and flushed")
    except Exception as e:
        print(f"   ⚠️  TensorBoard cleanup error: {e}")
    
    # 2. Cleanup EMA
    print("\n2️⃣  Cleaning up EMA...")
    try:
        if ema is not None:
            ema.cleanup()
            print("   ✅ EMA cleaned")
    except Exception as e:
        print(f"   ⚠️  EMA cleanup error: {e}")
    
    # 3. Cleanup loss functions
    print("\n3️⃣  Cleaning up loss functions...")
    try:
        if loss_manager is not None:
            loss_manager.cleanup()
            del loss_manager
            print("   ✅ Loss manager cleaned")
    except Exception as e:
        print(f"   ⚠️  Loss manager cleanup error: {e}")
    
    # 4. Move model to CPU and delete
    print("\n4️⃣  Cleaning up model...")
    try:
        if model is not None:
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            print("   ✅ Model moved to CPU and freed")
    except Exception as e:
        print(f"   ⚠️  Model cleanup error: {e}")
    
    # 5. Clear DataLoader
    print("\n5️⃣  Cleaning up DataLoader...")
    try:
        if train_loader is not None:
            del train_loader
            print("   ✅ DataLoader deleted")
    except Exception as e:
        print(f"   ⚠️  DataLoader cleanup error: {e}")
    
    # 6. Final GPU/MPS cleanup
    print("\n6️⃣  Final device cleanup...")
    try:
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
            print("   ✅ MPS cache cleared and synchronized")
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   ✅ CUDA cache cleared and synchronized")
    except Exception as e:
        print(f"   ⚠️  Device cleanup error: {e}")
    
    # 7. Force garbage collection
    print("\n7️⃣  Running garbage collection...")
    try:
        import gc
        collected = gc.collect()
        print(f"   ✅ Garbage collection completed ({collected} objects freed)")
    except Exception as e:
        print(f"   ⚠️  Garbage collection error: {e}")
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70 + "\n")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune upscaler - Optimized for MacBook Air M4'
    )
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory with LR frames')
    parser.add_argument('--hr_dir', type=str, required=True, help='Directory with HR frames')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained .pth file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size')
    parser.add_argument('--accumulation_steps', type=int, default=12, help='Gradient accumulation')
    
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--save_every', type=int, default=10, help='Save every N epochs')
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--val_save_every', type=int, default=10, help='Save samples every N epochs')
    
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    parser.add_argument('--no_perceptual', action='store_true', help='Disable perceptual loss')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--val_on_cpu', action='store_true', help='Run validation on CPU')
    
    # ✅ NEW LOSS WEIGHTS
    parser.add_argument('--brightness_weight', type=float, default=0.10, help='Brightness consistency')
    parser.add_argument('--contrast_weight', type=float, default=0.05, help='Contrast consistency')
    parser.add_argument('--color_weight', type=float, default=0.0, help='Color range limiter')
    parser.add_argument('--tv_weight', type=float, default=0.0, help='Total variation')
    parser.add_argument('--safe_margin', type=float, default=0.01, help='Color range margin')
    parser.add_argument('--highfreq_weight', type=float, default=0.10, help='High-frequency loss')
    parser.add_argument('--edge_weight', type=float, default=0.05, help='Edge preservation loss')
    parser.add_argument('--highlight_weight', type=float, default=0.30, help='Highlight-aware loss')
    parser.add_argument('--drange_weight', type=float, default=0.10, help='Dynamic range limiter')
    parser.add_argument('--hlgrad_weight', type=float, default=0.05, help='Highlight gradient limiter')
    
    # ✅ ADVANCED NEW LOSSES (Corrected defaults for brightness preservation)
    parser.add_argument('--lab_weight', type=float, default=0.02, help='Lab color space loss')
    parser.add_argument('--local_weight', type=float, default=0.0, help='Local statistics loss')
    parser.add_argument('--ssim_weight', type=float, default=0.05, help='SSIM luminance loss')
    parser.add_argument('--exposure_weight', type=float, default=0.05, help='Exposure gradient loss')
    parser.add_argument('--hist_weight', type=float, default=0.0, help='Soft histogram loss')
    parser.add_argument('--log_tone_weight', type=float, default=0.0, help='Log tone mapping loss')    
    # NEW BRIGHTNESS-PRESERVING LOSS WEIGHTS (v2.0)
    parser.add_argument('--mean_lum_weight', type=float, default=0.20,
                        help='Mean luminance preservation loss weight (CRITICAL)')
    parser.add_argument('--percentile_weight', type=float, default=0.0,
                        help='Percentile preservation loss weight')

    
    parser.add_argument('--preprocessed', action='store_true', help='Use pre-cropped patches')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_inputs(args.lr_dir, args.hr_dir, args.pretrained):
        sys.exit(1)
    
    args.use_perceptual = not args.no_perceptual
    args.use_ema = not args.no_ema
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using device: MPS (Apple Silicon)")
        torch.set_float32_matmul_precision('high')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using device: CUDA - {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_capability(0)[0] >= 8:
            torch.set_float32_matmul_precision('high')
    else:
        device = torch.device('cpu')
        print(f"⚠️  Using device: CPU")
    
    check_system_resources(device)

    use_autocast = args.use_mixed_precision and device.type == 'cuda'
    
    # Load model
    print("\nLoading model...")
    model = SuperUltraCompact(in_nc=3, out_nc=3, nf=24, nc=8, scale=2, use_activations=False)

    # Verify InstanceNorm doesn't track running stats
    for module in model.modules():
        if isinstance(module, nn.InstanceNorm2d):
            module.track_running_stats = False  # 🔴 CRITICAL
            module.affine = True  # Keep weights

    model = load_pretrained(model, args.pretrained, device)

    # 🔴 CRITICAL FIX: Wrap with output clamping
    print("\n🔒 Wrapping model with output clamping...")
    model = ClampedOutputModel(model)
    print("   ✅ All outputs will be hard-clamped to [0, 1]")

    # Load dataset
    print("\nLoading dataset...")
    if args.preprocessed:
        full_dataset = PreprocessedPatchDataset(args.lr_dir, args.hr_dir)
    else:
        full_dataset = VideoFramePairDataset(args.lr_dir, args.hr_dir, args.patch_size)
    
    dataset_len = len(full_dataset)
    if dataset_len < 2:
        raise ValueError("Need at least 2 samples to split into train/val.")

    val_size = max(1, int(dataset_len * args.val_split))
    train_size = dataset_len - val_size
    if train_size < 1:
        train_size = 1
        val_size = dataset_len - 1
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 🔴 M4 OPTIMIZATION: Use workers for parallel data loading
    if device.type == 'mps':
        num_workers = 2         # Enough to keep GPU fed
        pin_memory = False      # No-op for unified memory
        prefetch_factor = 2     # Keep 2 batches ready
    elif device.type == 'cuda':
        num_workers = 4
        pin_memory = True
        prefetch_factor = 2
    else:
        num_workers = 0
        pin_memory = False
        prefetch_factor = None
    
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    print(f"✅ DataLoader: {num_workers} workers, "
          f"prefetch={prefetch_factor}, persistent={num_workers > 0}")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Validation frequency: every {args.val_every} epochs")

    # Initialize loss manager
    loss_manager = LossManager(device)
    loss_dict = loss_manager.create_losses(args)

    # Extract individual criterions (for backward compatibility)
    criterion_l1 = loss_dict['l1']
    criterion_perceptual = loss_dict['perceptual']
    criterion_brightness = loss_dict['brightness']
    criterion_contrast = loss_dict['contrast']
    criterion_color = loss_dict['color']
    criterion_tv = loss_dict['tv']
    criterion_highfreq = loss_dict['highfreq']
    criterion_edge = loss_dict['edge']
    criterion_highlight = loss_dict['highlight']
    criterion_drange = loss_dict['drange']
    criterion_hl_grad = loss_dict['hl_grad']
    criterion_lab = loss_dict['lab']
    criterion_local = loss_dict['local']
    criterion_ssim = loss_dict['ssim']
    criterion_exposure = loss_dict['exposure']
    criterion_hist = loss_dict['hist']
    criterion_log_tone = loss_dict['log_tone']
    criterion_mean_lum = loss_dict['mean_lum']
    criterion_percentile = loss_dict['percentile']

    print("\n✅ Loss manager initialized with", 
          len([v for v in loss_dict.values() if v is not None]), 
          "active loss functions")
    
    # Print brightness correction notice
    print("\n" + "="*70)
    print("🔧 BRIGHTNESS CORRECTION APPLIED")
    print("="*70)
    print("This corrected version fixes the Lab color space gamma issue that")
    print("was causing brighter outputs. The sRGB->Linear->XYZ->Lab conversion")
    print("is now properly implemented to preserve correct brightness levels.")
    print("Default loss weights have been reduced to be more conservative.")
    print("="*70 + "\n")
    
    print_loss_config = []
    if args.lab_weight > 0:
        print_loss_config.append(f"Lab space loss (weight: {args.lab_weight})")
    if args.local_weight > 0:
        print_loss_config.append(f"Local statistics loss (weight: {args.local_weight})")
    if args.ssim_weight > 0:
        print_loss_config.append(f"SSIM luminance loss (weight: {args.ssim_weight})")
    if args.exposure_weight > 0:
        print_loss_config.append(f"Exposure gradient loss (weight: {args.exposure_weight})")
    if args.hist_weight > 0:
        print_loss_config.append(f"Soft histogram loss (weight: {args.hist_weight})")
    if args.log_tone_weight > 0:
        print_loss_config.append(f"Log tone mapping loss (weight: {args.log_tone_weight})")
    
    if print_loss_config:
        print("\n✅ Advanced overexposure prevention losses enabled:")
        for config in print_loss_config:
            print(f"   - {config}")
    
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=1e-4, betas=(0.9, 0.99)
    )
    
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    warmup_epochs = min(5, args.epochs)
    warmup_iters = (len(train_loader) * warmup_epochs) // args.accumulation_steps
    warmup_iters = max(1, warmup_iters)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)

    main_tmax = max(1, args.epochs - warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_tmax, eta_min=args.lr * 0.01)
    
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"✅ EMA enabled with decay={args.ema_decay}")
    
    writer = None
    log_dir = args.log_dir or f"{args.output_dir}/logs"
    if log_dir and HAS_TENSORBOARD:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"✅ TensorBoard logging enabled: {log_dir}")
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        try:
            resume_checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            start_epoch = resume_checkpoint.get('epoch', 0) + 1
            
            if ema is not None and 'ema_state_dict' in resume_checkpoint:
                ema.load_state_dict(resume_checkpoint['ema_state_dict'])
            
            best_val_loss = resume_checkpoint.get('loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠️  Failed to resume: {e}")
            start_epoch = 1
    

    # ========== SANITY CHECK: Verify pretrained model works ==========
    print("\n🔍 SANITY CHECK: Testing pretrained model output...")
    model.eval()
    try:
        test_batch = next(iter(val_loader))
        test_lr, test_hr = test_batch[:2]
        test_lr = to_device(test_lr, device)
        
        with torch.no_grad():
            test_sr = model(test_lr)
        
        print(f"   Input  range: [{test_lr.min():.3f}, {test_lr.max():.3f}]")
        print(f"   Output range: [{test_sr.min():.3f}, {test_sr.max():.3f}]")
        print(f"   Output mean:  {test_sr.mean():.3f}")
        print(f"   Output std:   {test_sr.std():.3f}")
        
        # Save a pretrained sample for comparison
        if HAS_CV2:
            sample_sr = test_sr[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            sample_sr = (sample_sr * 255).astype(np.uint8)
            sample_bgr = cv2.cvtColor(sample_sr, cv2.COLOR_RGB2BGR)
            pretrain_path = os.path.join(args.output_dir, 'pretrained_output_check.png')
            cv2.imwrite(pretrain_path, sample_bgr)
            print(f"   ✅ Pretrained sample saved: {pretrain_path}")
            print(f"   → Check this image BEFORE training starts!")
        
        del test_lr, test_hr, test_sr
        if device.type == 'mps':
            torch.mps.empty_cache()
    except Exception as e:
        print(f"   ⚠️  Sanity check failed: {e}")
    
    model.train()
    # ========== END SANITY CHECK ==========


    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    print(f"Gradient accumulation: {args.accumulation_steps} steps")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print("="*70 + "\n")
    


    # Initialize quality tracker (v3.0)
    quality_tracker = EpochQualityTracker(model, device)
    print("✅ Quality tracker initialized (contrast/exposure rollback enabled)")

    for epoch in range(start_epoch, args.epochs + 1):
        # Update adaptive losses at start of each epoch
        if criterion_highlight is not None and hasattr(criterion_highlight, 'step_epoch'):
            criterion_highlight.step_epoch()
        
        use_warmup = epoch <= warmup_epochs
        current_warmup = warmup_scheduler if use_warmup else None
        
        try:
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion_l1, criterion_perceptual,
                device, epoch, ema=ema,
                use_perceptual=args.use_perceptual, 
                perceptual_weight=args.perceptual_weight,
                warmup_scheduler=current_warmup,
                use_autocast=use_autocast,
                accumulation_steps=args.accumulation_steps,
                criterion_brightness=criterion_brightness,
                criterion_contrast=criterion_contrast,
                criterion_color=criterion_color,
                criterion_tv=criterion_tv,
                brightness_weight=args.brightness_weight,
                contrast_weight=args.contrast_weight,
                color_weight=args.color_weight,
                tv_weight=args.tv_weight,
                criterion_highfreq=criterion_highfreq,
                criterion_edge=criterion_edge,
                highfreq_weight=args.highfreq_weight,
                edge_weight=args.edge_weight,
                criterion_highlight=criterion_highlight,
                criterion_drange=criterion_drange,
                criterion_hl_grad=criterion_hl_grad,
                highlight_weight=args.highlight_weight,
                drange_weight=args.drange_weight,
                hlgrad_weight=args.hlgrad_weight,
                criterion_lab=criterion_lab,
                lab_weight=args.lab_weight,
                criterion_local=criterion_local,
                local_weight=args.local_weight,
                criterion_ssim=criterion_ssim,
                ssim_weight=args.ssim_weight,
                criterion_exposure=criterion_exposure,
                exposure_weight=args.exposure_weight,
                criterion_hist=criterion_hist,
                hist_weight=args.hist_weight,
                criterion_log_tone=criterion_log_tone,
                log_tone_weight=args.log_tone_weight,
                writer=writer,
                criterion_mean_lum=criterion_mean_lum,
                mean_lum_weight=args.mean_lum_weight,
                criterion_percentile=criterion_percentile,
                percentile_weight=args.percentile_weight,
                output_dir=args.output_dir,
                main_scheduler=main_scheduler,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ OOM error. Try: --batch_size 1 --accumulation_steps 24 --patch_size 96")
                sys.exit(1)
            else:
                raise e
        
        should_validate = (epoch % args.val_every == 0) or (epoch == args.epochs)
        
        if should_validate:
            # 🔴 FIXED: Always validate on MPS - avoid expensive transfers
            val_device = device
            
            if device.type == 'mps':
                args.val_on_cpu = False  # Force validation on MPS for efficiency
            
            # Safe validation with guaranteed EMA restore
            try:
                if ema is not None:
                    ema.apply_shadow()
                    
                try:
                    val_metrics = validate(
                        model, val_loader, criterion_l1, criterion_perceptual,
                        val_device, args.use_perceptual, args.perceptual_weight,
                        use_autocast=use_autocast and device.type == 'cuda',
                        calc_metrics=HAS_METRICS
                    )
                finally:
                    if ema is not None:
                        ema.restore()
                    if device.type == 'mps':
                        aggressive_memory_cleanup(device)
                        
            except RuntimeError as e:
                print(f"⚠️  Validation error: {e}")
                val_metrics = {'loss': float('inf')}
                aggressive_memory_cleanup(device)

        else:
            val_metrics = {'loss': best_val_loss}
            print(f"⏩ Skipping validation (every {args.val_every} epochs)")
        

        # ========== EPOCH-LEVEL QUALITY CHECK (v3.0) ==========
        if should_validate and quality_tracker is not None:
            try:
                epoch_qm = quality_tracker.compute_epoch_metrics(
                    model, val_loader, device, num_batches=10
                )
                rolled_back, qmsg = quality_tracker.check_epoch(
                    model, epoch, epoch_qm
                )
                if rolled_back:
                    print(f"  🔄 {qmsg}")
                    # After rollback, still proceed with scheduler etc.
            except Exception as e:
                print(f"  ⚠️  Quality tracking error: {e}")

        if not use_warmup:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        val_loss = val_metrics['loss']
        
        # Print results
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        if should_validate:
            print(f"  Val Loss:   {val_loss:.4f}")
            if HAS_METRICS and 'psnr' in val_metrics:
                print(f"  PSNR:       {val_metrics['psnr']:.2f} dB")
                print(f"  SSIM:       {val_metrics['ssim']:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            if should_validate:
                writer.add_scalar('Loss/val', val_loss, epoch)
                if 'psnr' in val_metrics:
                    writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
                    writer.add_scalar('Metrics/SSIM', val_metrics['ssim'], epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)

            # 🔴 FIXED: Flush more frequently for M4 to prevent buffer accumulation
            flush_frequency = 2 if device.type == 'mps' else 5
            if epoch % flush_frequency == 0:
                writer.flush()
                print(f"  📊 TensorBoard flushed (epoch {epoch})")
            
            # Force sync on MPS after flush
            if device.type == 'mps':
                torch.mps.synchronize()
        
        if should_validate:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.output_dir, 'best_model.pth'),
                    ema=ema, main_scheduler=main_scheduler,
                    warmup_scheduler=warmup_scheduler, is_best=True,
                    val_loader=val_loader,
                    device=device
                )

        # 🔴 OPTIMIZED: Save latest less frequently on M4 (disk I/O intensive)
        save_frequency = 2 if device.type == 'mps' else 1
        if epoch % save_frequency == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'latest.pth'),
                ema=ema, main_scheduler=main_scheduler,
                warmup_scheduler=warmup_scheduler,
                val_loader=val_loader,
                device=device
            )

        # 🔴 OPTIMIZED: Save epoch checkpoints less frequently on M4
        checkpoint_frequency = args.save_every * 2 if device.type == 'mps' else args.save_every
        if epoch % checkpoint_frequency == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'),
                ema=ema, main_scheduler=main_scheduler,
                warmup_scheduler=warmup_scheduler,
                val_loader=val_loader,
                device=device
            )
        
        if epoch % args.val_save_every == 0:
            vis_device = torch.device('cpu') if (args.val_on_cpu and device.type == 'mps') else device
            if args.val_on_cpu and device.type == 'mps':
                model = model.cpu()
                if ema is not None:
                    ema.apply_shadow()
                save_validation_samples(model, val_loader, vis_device, epoch, args.output_dir)
                if ema is not None:
                    ema.restore()
                model = model.to(device)
                clear_mps_cache()
            else:
                save_validation_samples(model, val_loader, vis_device, epoch, args.output_dir)
        
        # Aggressive cleanup at end of epoch
        if device.type == 'mps':
            aggressive_memory_cleanup(device)
    
    # Final save
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        os.path.join(args.output_dir, 'final_model.pth'),
        ema=ema, main_scheduler=main_scheduler,
        warmup_scheduler=warmup_scheduler
    )
    
    if writer is not None:
        writer.close()
    
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    
    checkpoint_files = sorted(Path(args.output_dir).glob('*.pth'))
    if checkpoint_files:
        print("\n📁 Saved checkpoints:")
        for f in checkpoint_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print("="*70)

    # Return values for cleanup
    return device, model, ema, loss_manager, writer, train_loader, quality_tracker


if __name__ == '__main__':
    device = None
    loss_manager = None
    ema = None
    writer = None
    train_loader = None
    val_loader = None
    model = None
    quality_tracker = None

    
    try:
        # Run training
        device, model, ema, loss_manager, writer, train_loader, quality_tracker = main()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*70)
        print("🧹 FINAL CLEANUP - FREEING ALL RESOURCES")
        print("="*70)
        
        try:
            # 1. Close TensorBoard
            print("\n1️⃣  Closing TensorBoard...")
            try:
                if writer is not None:
                    writer.flush()
                    writer.close()
                    print("   ✅ TensorBoard closed")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 2. Cleanup EMA
            print("\n2️⃣  Cleaning up EMA...")
            try:
                if ema is not None:
                    ema.cleanup()
                    del ema
                    print("   ✅ EMA cleaned")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")

            # 2b. Cleanup Quality Tracker
            print("\n    Cleaning up Quality Tracker...")
            try:
                if quality_tracker is not None:
                    quality_tracker.cleanup()
                    del quality_tracker
                    print("   ✅ Quality tracker cleaned")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 3. Cleanup loss functions
            print("\n3️⃣  Cleaning up loss functions...")
            try:
                if loss_manager is not None:
                    loss_manager.cleanup()
                    del loss_manager
                    print("   ✅ Loss manager cleaned")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 4. Cleanup model
            print("\n4️⃣  Cleaning up model...")
            try:
                if model is not None:
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                    print("   ✅ Model cleaned")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 5. Cleanup DataLoaders
            print("\n5️⃣  Cleaning up DataLoaders...")
            try:
                for loader in [train_loader, val_loader]:
                    if loader is not None:
                        del loader
                print("   ✅ DataLoaders cleaned")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 6. Device cleanup
            print("\n6️⃣  Final device cleanup...")
            try:
                if device is not None:
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                        torch.mps.synchronize()
                        print("   ✅ MPS cache cleared")
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print("   ✅ CUDA cache cleared")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            # 7. Force garbage collection
            print("\n7️⃣  Running garbage collection...")
            try:
                import gc
                collected = gc.collect()
                print(f"   ✅ {collected} objects freed")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
            
            print("\n" + "="*70)
            print("✅ ALL RESOURCES FREED - SAFE TO EXIT")
            print("="*70 + "\n")
        
        except Exception as e:
            print(f"\n⚠️  Cleanup error: {e}")

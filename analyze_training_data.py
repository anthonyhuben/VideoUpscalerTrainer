#!/usr/bin/env python3
"""
Training Data Analyzer v4.0 — Max-accuracy loss weight estimation.

Improvements over v2:
  - BCa (bias-corrected accelerated) bootstrap for more accurate CIs
  - Effect-size (Cohen's d) gating: only activates losses when the LR→HR
    difference is practically meaningful, not just statistically significant
  - Percentile-based metrics (p95, p99) for highlight/shadow analysis
  - Full-dataset analysis by default (sample_size=0)
  - Pairwise LR↑ vs HR metrics (PSNR/SSIM/MAE/RMSE/ΔE/correlation/MI)
  - CIE76 ΔE color difference for more accurate color weighting
  - Laplacian-variance texture metric for sharpness estimation
  - All suggested values rounded to 2 decimal places
  - Diagnostic warnings for data quality issues
  - Summary statistics table for quick sanity-checking
  - Deterministic seeding for reproducibility
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import random
import warnings

try:
    import cv2
except ImportError:
    print("❌ OpenCV (cv2) not installed. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    ssim = None
    psnr = None
    print("⚠️  scikit-image not installed. SSIM/PSNR metrics will be skipped.")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ensure_bgr(img):
    """Ensure OpenCV image is 3-channel BGR uint8."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _entropy_from_hist(hist):
    """Shannon entropy from a normalized histogram."""
    hist = hist.astype(np.float64)
    hist = hist / (np.sum(hist) + 1e-12)
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def _mutual_information(x, y, bins=64):
    """Mutual information between two flattened arrays in [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    h2d, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
    pxy = h2d / (np.sum(h2d) + 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px_py[nz] + 1e-12))))


def calculate_metrics(img):
    """
    Compute per-image metrics.  Input is BGR uint8 (OpenCV default).
    Returns dict of scalar float64 values.
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float64) / 255.0

    # Channels (OpenCV is BGR)
    b_ch, g_ch, r_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    # ITU-R BT.709 luminance (matches training code)
    lum = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch

    # --- Texture / sharpness ---
    lap = cv2.Laplacian(lum, cv2.CV_64F)
    sharpness = np.var(lap)

    gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3)
    edge_energy = np.mean(np.sqrt(gx ** 2 + gy ** 2))

    # --- Dynamic range ---
    high_pix = np.mean(lum > 0.90)
    low_pix  = np.mean(lum < 0.05)

    # --- Percentiles (more robust than min/max) ---
    lum_flat = lum.ravel()
    p01 = np.percentile(lum_flat, 1)
    p05 = np.percentile(lum_flat, 5)
    p50 = np.percentile(lum_flat, 50)
    p95 = np.percentile(lum_flat, 95)
    p99 = np.percentile(lum_flat, 99)

    # --- Local contrast (32×32 grid variance) ---
    small = cv2.resize(lum, (32, 32), interpolation=cv2.INTER_AREA)
    local_var = np.var(small)

    # --- Entropy (texture complexity) ---
    hist = np.histogram(lum_flat, bins=256, range=(0, 1))[0]
    entropy = _entropy_from_hist(hist)

    # --- Clipping (near-black/near-white) ---
    clip_high = np.mean(lum > 0.98)
    clip_low = np.mean(lum < 0.02)

    # --- Color balance ---
    mean_r, mean_g, mean_b = np.mean(r_ch), np.mean(g_ch), np.mean(b_ch)
    color_spread = max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b)

    return {
        'mean_lum':     np.mean(lum),
        'std_lum':      np.std(lum),
        'mean_r':       mean_r,
        'mean_g':       mean_g,
        'mean_b':       mean_b,
        'color_spread': color_spread,
        'sharpness':    sharpness,
        'edge_energy':  edge_energy,
        'high_pixels':  high_pix,
        'low_pixels':   low_pix,
        'local_var':    local_var,
        'entropy':      entropy,
        'clip_high':    clip_high,
        'clip_low':     clip_low,
        'min_val':      float(np.min(lum)),
        'max_val':      float(np.max(lum)),
        'p01':          p01,
        'p05':          p05,
        'p50':          p50,
        'p95':          p95,
        'p99':          p99,
    }


def calculate_pair_metrics(lr_img, hr_img, mi_bins=64):
    """
    Compute pairwise metrics between upscaled LR and HR.
    Returns dict of scalar float64 values.
    """
    # Ensure same size
    if lr_img.shape[:2] != hr_img.shape[:2]:
        lr_img = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]),
                            interpolation=cv2.INTER_CUBIC)

    # Float images in [0, 1]
    lr_f = lr_img.astype(np.float64) / 255.0
    hr_f = hr_img.astype(np.float64) / 255.0

    # Luminance (BT.709)
    lr_lum = 0.2126 * lr_f[:, :, 2] + 0.7152 * lr_f[:, :, 1] + 0.0722 * lr_f[:, :, 0]
    hr_lum = 0.2126 * hr_f[:, :, 2] + 0.7152 * hr_f[:, :, 1] + 0.0722 * hr_f[:, :, 0]

    diff = hr_lum - lr_lum
    mae_lum = float(np.mean(np.abs(diff)))
    rmse_lum = float(np.sqrt(np.mean(diff ** 2)))

    # Correlation (luminance)
    lr_flat = lr_lum.ravel()
    hr_flat = hr_lum.ravel()
    if np.std(lr_flat) > 1e-12 and np.std(hr_flat) > 1e-12:
        corr_lum = float(np.corrcoef(lr_flat, hr_flat)[0, 1])
    else:
        corr_lum = 0.0

    # Mutual information (luminance)
    mi_lum = _mutual_information(lr_flat, hr_flat, bins=mi_bins)

    # CIE76 DeltaE (convert to Lab)
    lr_lab = cv2.cvtColor(lr_img, cv2.COLOR_BGR2LAB).astype(np.float64)
    hr_lab = cv2.cvtColor(hr_img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Convert to standard Lab ranges: L in [0,100], a/b centered at 0
    lr_L = lr_lab[:, :, 0] * (100.0 / 255.0)
    hr_L = hr_lab[:, :, 0] * (100.0 / 255.0)
    lr_a = lr_lab[:, :, 1] - 128.0
    hr_a = hr_lab[:, :, 1] - 128.0
    lr_b = lr_lab[:, :, 2] - 128.0
    hr_b = hr_lab[:, :, 2] - 128.0
    delta_e = float(np.mean(np.sqrt((hr_L - lr_L) ** 2 + (hr_a - lr_a) ** 2 + (hr_b - lr_b) ** 2)))

    metrics = {
        'mae_lum': mae_lum,
        'rmse_lum': rmse_lum,
        'corr_lum': corr_lum,
        'mi_lum': mi_lum,
        'delta_e': delta_e,
    }

    if HAS_SKIMAGE:
        metrics['ssim_lr_hr'] = float(ssim(hr_f, lr_f, data_range=1.0, channel_axis=2))

    # PSNR (manual to avoid divide-by-zero warnings)
    mse = np.mean((hr_f - lr_f) ** 2)
    if mse <= 1e-12:
        psnr_val = 100.0  # effectively "infinite" for perfect matches
    else:
        psnr_val = 10.0 * np.log10(1.0 / mse)
    metrics['psnr_lr_hr'] = float(psnr_val)

    return metrics


# ---------------------------------------------------------------------------
# Bootstrap (BCa)
# ---------------------------------------------------------------------------

def _bca_bootstrap(data, n_boot=10000, ci=95, rng=None):
    """
    Bias-corrected and accelerated (BCa) bootstrap.
    Returns (mean, ci_low, ci_high).
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 3:
        m = float(np.mean(data)) if n > 0 else 0.0
        return m, m, m

    observed = np.mean(data)

    # --- bootstrap distribution ---
    rng = rng or np.random.default_rng(seed=42)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(data, size=n, replace=True))

    # --- bias correction ---
    z0 = _norm_ppf(np.mean(boot_means < observed))

    # --- acceleration (jackknife) ---
    # Mean jackknife can be computed in O(n) exactly.
    total = np.sum(data)
    jack = (total - data) / (n - 1)
    jack_mean = float(np.mean(jack))
    diff = jack_mean - jack
    a = np.sum(diff ** 3) / (6.0 * (np.sum(diff ** 2) ** 1.5 + 1e-12))

    # --- adjusted percentiles ---
    alpha = (100 - ci) / 200.0
    zl, zu = _norm_ppf(alpha), _norm_ppf(1 - alpha)

    a1 = _norm_cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl) + 1e-12))
    a2 = _norm_cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu) + 1e-12))

    ci_low  = float(np.percentile(boot_means, 100 * max(0, min(1, a1))))
    ci_high = float(np.percentile(boot_means, 100 * max(0, min(1, a2))))
    robust_mean = float(np.mean(boot_means))

    return robust_mean, ci_low, ci_high


def _norm_ppf(p):
    """Inverse normal CDF (probit) via rational approximation."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    # Beasley-Springer-Moro algorithm
    from math import log, sqrt
    a = p - 0.5
    if abs(a) < 0.42:
        r = a * a
        return a * ((((-25.44106049637 * r + 41.39119773534) * r
                       - 18.61500062529) * r + 2.50662823884)
                    / ((((3.13082909833 * r - 21.06224101826) * r
                         + 23.08336743743) * r - 8.47351093090) * r + 1))
    else:
        r = p if a <= 0 else 1 - p
        r = sqrt(-2 * log(r))
        result = (((0.3374754822726147 * r + 0.9761690190917186) * r
                    - 1.5614734589656076) * r - 0.3989422804014327)
        result /= ((0.01033808670612024 * r + 0.2015871662874588) * r + 1)
        return result if a >= 0 else -result


def _norm_cdf(x):
    """Standard-normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(diffs):
    """Cohen's d for paired differences (effect size)."""
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-12))


# ---------------------------------------------------------------------------
# Weight suggestion engine
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def r2(v):
    """Round to 2 decimals."""
    return round(float(v), 2)


def analyze_and_suggest(stats_diff, stats_hr, stats_lr, effect_sizes, stats_pair=None):
    """
    Produce weight suggestions.  All values are rounded to 2 decimals.

    stats_diff / stats_hr / stats_lr: {key: (mean, ci_lo, ci_hi)}
    effect_sizes: {key: float}  — Cohen's d for the paired differences
    """
    suggestions = {}

    def set_param(key, val, val_low, val_high, reason):
        suggestions[key] = {
            'value':  r2(val),
            'range':  f"{r2(val_low)}-{r2(val_high)}",
            'reason': reason,
        }

    # Helpers
    def abs_mean(k):
        return abs(stats_diff[k][0])

    def hr_mean(k):
        return stats_hr[k][0]

    def lr_mean(k):
        return stats_lr[k][0]

    def es(k):
        return abs(effect_sizes.get(k, 0.0))

    def pair_mean(k, default=None):
        if stats_pair and k in stats_pair:
            return stats_pair[k][0]
        return default

    # =====================================================================
    # 1. PERCEPTUAL WEIGHT
    #    High when HR is clean; reduced when HR itself is noisy (sharpness
    #    proxy).  Perceptual loss amplifies noise if HR is noisy.
    # =====================================================================
    hr_sharp = hr_mean('sharpness')
    if hr_sharp > 0.03:
        pv = clamp(1.0 - (hr_sharp - 0.03) / 0.04, 0.10, 1.0)
    else:
        pv = 1.0
    set_param('perceptual_weight', pv, pv * 0.85, min(pv * 1.15, 1.0),
              f"HR noise proxy {hr_sharp:.4f}")

    # =====================================================================
    # 2. SSIM WEIGHT
    #    Driven by structural (edge) delta between LR and HR.
    # =====================================================================
    ed = abs_mean('edge_energy')
    ed_es = es('edge_energy')
    if ed_es > 0.3:  # meaningful effect
        sv = clamp(0.10 + ed * 3.0, 0.10, 0.60)
    else:
        sv = 0.10
    ssim_lr = pair_mean('ssim_lr_hr')
    if ssim_lr is not None:
        # Boost SSIM weight when LR↑ is structurally far from HR
        ssim_gap = max(0.0, 0.90 - ssim_lr)
        sv = clamp(sv + ssim_gap * 0.80, 0.10, 0.80)
        reason = f"Edge Δ={ed:.4f}, d={ed_es:.2f}, SSIM={ssim_lr:.3f}"
    else:
        reason = f"Edge Δ={ed:.4f}, d={ed_es:.2f}"
    set_param('ssim_weight', sv, sv * 0.80, sv * 1.20, reason)

    # =====================================================================
    # 3. LAB WEIGHT
    #    Driven by average per-channel color shift (RGB mean diffs).
    # =====================================================================
    avg_color_diff = (abs_mean('mean_r') + abs_mean('mean_g') + abs_mean('mean_b')) / 3.0
    delta_e = pair_mean('delta_e')
    if delta_e is not None:
        if delta_e > 1.5:
            lv = clamp(0.05 + delta_e * 0.03, 0.05, 0.60)
        else:
            lv = 0.05
        reason = f"ΔE {delta_e:.2f} (CIE76)"
    else:
        if avg_color_diff > 0.005:
            lv = clamp(0.05 + avg_color_diff * 6.0, 0.05, 0.60)
        else:
            lv = 0.05
        reason = f"Color shift {avg_color_diff:.4f}"
    set_param('lab_weight', lv, lv * 0.80, lv * 1.20, reason)

    # =====================================================================
    # 4. BRIGHTNESS WEIGHT
    #    Driven by mean-luminance drift.
    # =====================================================================
    lum_d = abs_mean('mean_lum')
    lum_es = es('mean_lum')
    if lum_es > 0.2:
        bv = clamp(0.05 + lum_d * 5.0, 0.05, 0.50)
    else:
        bv = 0.05
    set_param('brightness_weight', bv, bv * 0.80, bv * 1.20,
              f"Lum Δ={lum_d:.4f}, d={lum_es:.2f}")

    # =====================================================================
    # 5. HIGHLIGHT WEIGHT
    #    Proportional to fraction of HR pixels in highlight zone.
    # =====================================================================
    hr_hi = hr_mean('high_pixels')
    hv = clamp(0.10 + hr_hi * 4.0, 0.10, 0.80)
    set_param('highlight_weight', hv, hv * 0.85, hv * 1.15,
              f"{hr_hi * 100:.1f}% highlight pixels")

    # =====================================================================
    # EXPERIMENTAL / AUXILIARY LOSSES
    # =====================================================================

    # Contrast
    con_d = abs_mean('std_lum')
    con_es = es('std_lum')
    if con_es > 0.3 and con_d > 0.005:
        cv_ = clamp(0.02 + con_d * 3.0, 0.02, 0.20)
    else:
        cv_ = 0.0
    set_param('contrast_weight', cv_, cv_ * 0.80, cv_ * 1.20,
              f"Contrast Δ={con_d:.4f}, d={con_es:.2f}" if cv_ > 0 else "Stable")

    # Color range
    if avg_color_diff > 0.01:
        clr = clamp(avg_color_diff * 2.5, 0.0, 0.15)
    else:
        clr = 0.0
    set_param('color_weight', clr, clr * 0.80, clr * 1.20,
              f"Color diff {avg_color_diff:.4f}" if clr > 0 else "Not needed")

    # TV (noise suppression)
    if hr_sharp > 0.015:
        tvv = clamp((hr_sharp - 0.015) * 1.0, 0.0, 0.03)
    else:
        tvv = 0.0
    set_param('tv_weight', tvv, tvv * 0.80, tvv * 1.20,
              f"HR noise {hr_sharp:.4f}" if tvv > 0 else "Clean")

    # Safe margin
    hr_p99 = hr_mean('p99')
    hr_p01 = hr_mean('p01')
    if hr_p99 > 0.98 or hr_p01 < 0.01:
        set_param('safe_margin', 0.02, 0.01, 0.03, "Near-clipping detected")
    else:
        set_param('safe_margin', 0.01, 0.00, 0.02, "Range safe")

    # Local statistics
    loc_d = abs_mean('local_var')
    loc_es = es('local_var')
    if loc_es > 0.3 and loc_d > 0.001:
        locv = clamp(0.05 + loc_d * 15.0, 0.05, 0.20)
    else:
        locv = 0.0
    set_param('local_weight', locv, locv * 0.80, locv * 1.20,
              f"Local var Δ={loc_d:.4f}, d={loc_es:.2f}" if locv > 0 else "Stable")

    # Exposure gradient (assist brightness if brightness is high)
    if bv > 0.15:
        exv = r2(bv * 0.35)
    else:
        exv = 0.0
    set_param('exposure_weight', exv, exv * 0.85, exv * 1.15,
              "Assisting brightness" if exv > 0 else "Not needed")

    # Histogram / percentile (assist contrast if active)
    if cv_ > 0.04:
        histv = r2(cv_ * 0.40)
        percv = r2(cv_ * 0.35)
    else:
        histv = 0.0
        percv = 0.0
    set_param('hist_weight', histv, histv * 0.80, histv * 1.20,
              "Assisting contrast" if histv > 0 else "Not needed")
    set_param('percentile_weight', percv, percv * 0.80, percv * 1.20,
              "Assisting contrast" if percv > 0 else "Not needed")

    # Log tone mapping (shadow preservation)
    hr_lo = hr_mean('low_pixels')
    if hr_lo > 0.03:
        ltv = clamp(hr_lo * 1.5, 0.0, 0.15)
    else:
        ltv = 0.0
    set_param('log_tone_weight', ltv, ltv * 0.80, ltv * 1.20,
              f"{hr_lo * 100:.1f}% shadow pixels" if ltv > 0 else "Few shadows")

    # Dynamic range & highlight gradient (tied to highlight weight)
    if hv > 0.15:
        drv = r2(hv * 0.35)
        hlgv = r2(hv * 0.18)
    else:
        drv = 0.0
        hlgv = 0.0
    set_param('drange_weight', drv, drv * 0.80, drv * 1.20,
              "Highlight protection" if drv > 0 else "Not needed")
    set_param('hlgrad_weight', hlgv, hlgv * 0.80, hlgv * 1.20,
              "Gradient protection" if hlgv > 0 else "Not needed")

    # High-frequency & edge preservation
    lr_sharp = lr_mean('sharpness')
    ratio = hr_sharp / (lr_sharp + 1e-8)
    if ratio > 1.15:
        hfv = clamp((ratio - 1.15) * 0.15, 0.0, 0.30)
        edv = r2(hfv * 0.50)
    else:
        hfv = 0.0
        edv = 0.0
    set_param('highfreq_weight', hfv, hfv * 0.80, hfv * 1.20,
              f"HR/LR sharpness ratio {ratio:.2f}" if hfv > 0 else "Similar sharpness")
    set_param('edge_weight', edv, edv * 0.80, edv * 1.20,
              "Edge assist" if edv > 0 else "Similar sharpness")

    # Mean luminance preservation
    if lum_d > 0.01:
        mlv = clamp(0.10 + lum_d * 8.0, 0.10, 1.00)
    else:
        mlv = 0.10
    set_param('mean_lum_weight', mlv, mlv * 0.80, mlv * 1.20,
              f"Lum drift {lum_d:.4f}")

    return suggestions


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(raw_lr, raw_hr, raw_diff, n_pairs):
    """Print data quality warnings."""
    issues = []

    # 1. Sample size
    if n_pairs < 50:
        issues.append(f"⚠️  Very small sample ({n_pairs} pairs). CIs will be wide.")
    elif n_pairs < 200:
        issues.append(f"ℹ️  Moderate sample ({n_pairs} pairs). Consider ≥500 for tighter CIs.")

    # 2. Clipping
    if 'clip_high' in raw_hr:
        mean_clip_high = float(np.mean(raw_hr['clip_high']))
        if mean_clip_high > 0.02:
            issues.append(f"⚠️  Avg {mean_clip_high*100:.1f}% of HR pixels are near-clipped (>0.98).")
    else:
        hr_maxes = np.array(raw_hr['max_val'])
        clip_frac = np.mean(hr_maxes > 0.99)
        if clip_frac > 0.10:
            issues.append(f"⚠️  {clip_frac*100:.0f}% of HR images have near-clipped highlights (max > 0.99).")

    # 3. Very dark frames
    hr_means = np.array(raw_hr['mean_lum'])
    dark_frac = np.mean(hr_means < 0.10)
    if dark_frac > 0.15:
        issues.append(f"⚠️  {dark_frac*100:.0f}% of HR images are very dark (mean lum < 0.10).")

    if 'clip_low' in raw_hr:
        mean_clip_low = float(np.mean(raw_hr['clip_low']))
        if mean_clip_low > 0.02:
            issues.append(f"⚠️  Avg {mean_clip_low*100:.1f}% of HR pixels are near-black (<0.02).")

    # 4. Size mismatch warning (already handled via resize, but flag it)
    lr_sharps = np.array(raw_lr['sharpness'])
    if np.std(lr_sharps) > 0.03:
        issues.append("ℹ️  High variance in LR sharpness — dataset may mix different source qualities.")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze training data and suggest loss weights (v3.0)')
    parser.add_argument('--lr_dir', type=str, default='./data/lr_frames')
    parser.add_argument('--hr_dir', type=str, default='./data/hr_frames')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Number of image pairs to sample (0 = use all)')
    parser.add_argument('--bootstrap_iters', type=int, default=20000,
                        help='BCa bootstrap iterations (higher = more accurate)')
    parser.add_argument('--bootstrap_ci', type=int, default=95,
                        help='Confidence interval percent (default: 95)')
    parser.add_argument('--mi_bins', type=int, default=64,
                        help='Bins for mutual information (default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    lr_path = Path(args.lr_dir)
    hr_path = Path(args.hr_dir)

    if not lr_path.exists() or not hr_path.exists():
        print("❌ Error: LR or HR directory not found.")
        sys.exit(1)

    # --- Find pairs ---
    exts = {'.png', '.jpg', '.jpeg'}
    lr_files = sorted(f.name for f in lr_path.iterdir() if f.suffix.lower() in exts)
    hr_names = set(f.name for f in hr_path.iterdir() if f.suffix.lower() in exts)
    common = [f for f in lr_files if f in hr_names]

    if not common:
        print("❌ Error: No matching filename pairs found.")
        sys.exit(1)

    if args.sample_size <= 0 or args.sample_size >= len(common):
        samples = list(common)
    else:
        samples = random.sample(common, args.sample_size)
    sample_count = len(samples)

    print(f"📊 Dataset: {len(common)} pairs available, sampling {sample_count}")
    print(f"   LR: {lr_path}")
    print(f"   HR: {hr_path}")
    print(f"   Seed: {args.seed}")
    print()

    # --- Collect metrics ---
    KEYS = list(calculate_metrics(np.zeros((8, 8, 3), dtype=np.uint8)).keys())
    PAIR_KEYS = list(calculate_pair_metrics(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
        mi_bins=args.mi_bins
    ).keys())

    raw_lr   = {k: [] for k in KEYS}
    raw_hr   = {k: [] for k in KEYS}
    raw_diff = {k: [] for k in KEYS}
    raw_pair = {k: [] for k in PAIR_KEYS}
    skipped  = 0
    resize_count = 0
    scale_ratios = []

    for fname in tqdm(samples, desc="Analyzing images"):
        l_img = _ensure_bgr(cv2.imread(str(lr_path / fname)))
        h_img = _ensure_bgr(cv2.imread(str(hr_path / fname)))
        if l_img is None or h_img is None:
            skipped += 1
            continue

        # Resize LR to HR dimensions for fair comparison
        if l_img.shape[:2] != h_img.shape[:2]:
            resize_count += 1
            scale_ratios.append((
                h_img.shape[1] / max(1, l_img.shape[1]),
                h_img.shape[0] / max(1, l_img.shape[0]),
            ))
            l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]),
                               interpolation=cv2.INTER_CUBIC)
        try:
            lm = calculate_metrics(l_img)
            hm = calculate_metrics(h_img)
            pm = calculate_pair_metrics(l_img, h_img, mi_bins=args.mi_bins)
            for k in KEYS:
                raw_lr[k].append(lm[k])
                raw_hr[k].append(hm[k])
                raw_diff[k].append(hm[k] - lm[k])
            for k in PAIR_KEYS:
                raw_pair[k].append(pm[k])
        except Exception:
            skipped += 1

    n_valid = len(raw_lr['mean_lum'])
    if n_valid == 0:
        print("❌ No valid image pairs processed.")
        sys.exit(1)

    if skipped > 0:
        print(f"⚠️  Skipped {skipped}/{sample_count} images (unreadable or errors)")
    print(f"✅ Processed {n_valid} pairs\n")
    if resize_count > 0 and scale_ratios:
        sr = np.array(scale_ratios, dtype=np.float64)
        mean_scale = np.mean(sr, axis=0)
        std_scale = np.std(sr, axis=0)
        print(f"ℹ️  Resized {resize_count}/{sample_count} LR images to match HR dimensions")
        print(f"   Scale factors (w,h): mean=({mean_scale[0]:.2f}, {mean_scale[1]:.2f}), "
              f"std=({std_scale[0]:.2f}, {std_scale[1]:.2f})")
        if std_scale[0] > 0.10 or std_scale[1] > 0.10:
            print("⚠️  Large variation in LR→HR scaling ratios — dataset may mix sources.")
        print()

    # --- Diagnostics ---
    issues = run_diagnostics(raw_lr, raw_hr, raw_diff, n_valid)
    if issues:
        print("🔍 DATA QUALITY DIAGNOSTICS:")
        for issue in issues:
            print(f"   {issue}")
        print()

    # --- Bootstrap ---
    print(f"📈 Computing BCa bootstrap confidence intervals ({args.bootstrap_iters} iterations)...")
    stats_diff = {}
    stats_hr   = {}
    stats_lr   = {}
    stats_pair = {}
    effect_sizes = {}

    rng = np.random.default_rng(args.seed)

    for k in tqdm(KEYS, desc="Bootstrapping"):
        stats_diff[k] = _bca_bootstrap(raw_diff[k], n_boot=args.bootstrap_iters,
                                       ci=args.bootstrap_ci, rng=rng)
        stats_hr[k]   = _bca_bootstrap(raw_hr[k], n_boot=args.bootstrap_iters,
                                       ci=args.bootstrap_ci, rng=rng)
        stats_lr[k]   = _bca_bootstrap(raw_lr[k], n_boot=args.bootstrap_iters,
                                       ci=args.bootstrap_ci, rng=rng)
        effect_sizes[k] = cohens_d(raw_diff[k])

    for k in tqdm(PAIR_KEYS, desc="Bootstrapping (pair metrics)"):
        stats_pair[k] = _bca_bootstrap(raw_pair[k], n_boot=args.bootstrap_iters,
                                       ci=args.bootstrap_ci, rng=rng)

    # --- Summary statistics table ---
    print("\n" + "=" * 105)
    print("📋 RAW STATISTICS SUMMARY")
    print("=" * 105)
    print(f"{'Metric':<16} | {'LR Mean':>9} | {'HR Mean':>9} | {'Diff':>9} | {'95% CI':>17} | {'Cohen d':>8} | {'Interp.'}")
    print("-" * 105)
    for k in KEYS:
        lm = stats_lr[k][0]
        hm = stats_hr[k][0]
        dm, dl, dh = stats_diff[k]
        d = effect_sizes[k]
        # Interpret effect size
        ad = abs(d)
        if ad < 0.2:
            interp = "negligible"
        elif ad < 0.5:
            interp = "small"
        elif ad < 0.8:
            interp = "medium"
        else:
            interp = "LARGE"
        print(f"{k:<16} | {lm:9.5f} | {hm:9.5f} | {dm:+9.5f} | [{dl:+.4f}, {dh:+.4f}] | {d:+8.3f} | {interp}")
    print("=" * 105)

    # --- Pairwise metrics table ---
    if PAIR_KEYS:
        print("\n" + "=" * 80)
        print("🔗 PAIRWISE LR↑ vs HR METRICS SUMMARY")
        print("=" * 80)
        print(f"{'Metric':<18} | {'Mean':>10} | {'95% CI':>17}")
        print("-" * 80)
        for k in PAIR_KEYS:
            m, lo, hi = stats_pair[k]
            print(f"{k:<18} | {m:10.5f} | [{lo:+.4f}, {hi:+.4f}]")
        print("=" * 80)

    # --- Weight suggestions ---
    suggestions = analyze_and_suggest(stats_diff, stats_hr, stats_lr, effect_sizes, stats_pair)

    conservative_keys = [
        'perceptual_weight', 'ssim_weight', 'lab_weight',
        'brightness_weight', 'highlight_weight',
    ]
    experimental_keys = [
        'contrast_weight', 'color_weight', 'tv_weight', 'safe_margin',
        'local_weight', 'exposure_weight', 'hist_weight', 'log_tone_weight',
        'drange_weight', 'hlgrad_weight', 'highfreq_weight', 'edge_weight',
        'mean_lum_weight', 'percentile_weight',
    ]

    print("\n" + "=" * 100)
    print(f"🎯 SUGGESTED LOSS WEIGHTS (n={n_valid}, 2-decimal precision)")
    print("=" * 100)
    print(f"{'PARAMETER':<22} | {'VALUE':>6} | {'95% CI RANGE':>13} | {'REASONING'}")
    print("-" * 100)

    def print_section(keys, title):
        print(f"\n  ── {title} ──")
        for k in keys:
            s = suggestions[k]
            print(f"  {k:<22} | {s['value']:6.2f} | {s['range']:>13} | {s['reason']}")

    print_section(conservative_keys, "CONSERVATIVE (always active)")
    print_section(experimental_keys, "EXPERIMENTAL (data-driven)")

    # --- Shell copy-paste block ---
    print("\n" + "=" * 100)
    print("📋 COPY/PASTE BLOCK FOR train_mac.sh")
    print("=" * 100)

    print("  `# Conservative loss weights (tested and stable)` \\")
    for k in conservative_keys:
        print(f"  --{k} {suggestions[k]['value']:.2f} \\")

    print("  \\")
    print("  `# Experimental losses` \\")
    for k in experimental_keys:
        print(f"  --{k} {suggestions[k]['value']:.2f} \\")

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()

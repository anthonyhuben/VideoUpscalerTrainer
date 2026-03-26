#!/usr/bin/env python3
"""
Multi-core detail gain validation with spectral and gradient analysis
Measures actual detail preservation vs. upscaling artifacts
"""

import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

# Handle older PIL versions
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS

# Optional scipy for better filters
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found, using numpy-only filters (install scipy for better accuracy)")


def extract_group_prefix(filename, parts=2):
    """Extract group prefix from filename."""
    stem = Path(filename).stem
    segments = stem.split('_')
    if len(segments) >= parts:
        return '_'.join(segments[:parts])
    return stem


def calculate_detail_metrics(img_array):
    """
    Calculate detail metrics from grayscale array (0-1 float).
    Returns dict of sharpness/texture measures.
    """
    metrics = {}
    h, w = img_array.shape
    
    # 1. Laplacian Variance (classic sharpness metric)
    if HAS_SCIPY:
        laplacian = ndimage.laplace(img_array)
    else:
        # Simple Laplacian kernel with numpy
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        from scipy.signal import convolve2d
        laplacian = convolve2d(img_array, kernel, mode='same', boundary='fill')
    metrics['laplacian_var'] = np.var(laplacian)
    
    # 2. High Frequency Energy Ratio (FFT-based)
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Create high-frequency mask (outside center 20%)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    center_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    radius = min(h, w) * 0.2
    high_freq_mask = center_dist > radius
    
    total_energy = np.sum(magnitude) + 1e-8
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    metrics['high_freq_ratio'] = high_freq_energy / total_energy
    metrics['spectral_centroid'] = np.sum(magnitude * center_dist) / total_energy
    
    # 3. Gradient Statistics (Edge strength)
    if HAS_SCIPY:
        dx = ndimage.sobel(img_array, axis=1)
        dy = ndimage.sobel(img_array, axis=0)
    else:
        # Simple finite differences
        dx = np.diff(img_array, axis=1, append=img_array[:, -1:])
        dy = np.diff(img_array, axis=0, append=img_array[-1:, :])
    
    gradient = np.hypot(dx, dy)
    metrics['gradient_mean'] = np.mean(gradient)
    metrics['gradient_std'] = np.std(gradient)
    metrics['gradient_max'] = np.max(gradient)
    
    # 4. Edge Density (% of pixels with strong edges)
    edge_threshold = metrics['gradient_mean'] + metrics['gradient_std']
    metrics['edge_density'] = np.sum(gradient > edge_threshold) / (h * w)
    
    # 5. Local Contrast (std in 8x8 blocks, averaged)
    block_size = 8
    if h >= block_size and w >= block_size:
        # Reshape into blocks and calc std for each
        h_blocks = h // block_size
        w_blocks = w // block_size
        blocked = img_array[:h_blocks*block_size, :w_blocks*block_size]
        blocked = blocked.reshape(h_blocks, block_size, w_blocks, block_size)
        local_stds = np.std(blocked, axis=(1, 3))
        metrics['local_contrast'] = np.mean(local_stds)
    else:
        metrics['local_contrast'] = np.std(img_array)
    
    # 6. Information Entropy (Shannon)
    hist, _ = np.histogram(img_array, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    metrics['entropy'] = -np.sum(hist * np.log2(hist + 1e-8))
    
    # 7. Texture Complexity (Laplacian histogram spread)
    metrics['texture_score'] = np.percentile(np.abs(laplacian), 95)
    
    return metrics


def analyze_detail_gain(args):
    """Worker function to compare detail between LR and HR."""
    lr_path, hr_path, max_size = args
    
    try:
        if not os.path.exists(lr_path) or not os.path.exists(hr_path):
            return {'filename': Path(lr_path).name, 'error': 'File not found', 'status': 'ERROR'}
        
        with Image.open(lr_path) as lr_img, Image.open(hr_path) as hr_img:
            # Convert to grayscale for detail analysis
            lr_gray = np.array(lr_img.convert('L')).astype(np.float32) / 255.0
            hr_gray = np.array(hr_img.convert('L')).astype(np.float32) / 255.0
            
            # Optional resize for performance
            if max_size and max_size > 0:
                if lr_gray.shape[0] > max_size or lr_gray.shape[1] > max_size:
                    lr_pil = Image.fromarray((lr_gray * 255).astype(np.uint8))
                    lr_pil.thumbnail((max_size, max_size), RESAMPLE_FILTER)
                    lr_gray = np.array(lr_pil).astype(np.float32) / 255.0
                    
                    hr_pil = Image.fromarray((hr_gray * 255).astype(np.uint8))
                    hr_pil.thumbnail((max_size, max_size), RESAMPLE_FILTER)
                    hr_gray = np.array(hr_pil).astype(np.float32) / 255.0
            
            # Downsample HR to LR dimensions for fair comparison
            # This tells us if HR actually has more detail per unit area
            hr_resized = np.array(
                Image.fromarray((hr_gray * 255).astype(np.uint8))
                .resize((lr_gray.shape[1], lr_gray.shape[0]), RESAMPLE_FILTER)
            ).astype(np.float32) / 255.0
            
            # Calculate metrics for both
            lr_metrics = calculate_detail_metrics(lr_gray)
            hr_metrics = calculate_detail_metrics(hr_resized)
            
            # Calculate gains and ratios
            epsilon = 1e-8
            gains = {}
            ratios = {}
            
            for key in lr_metrics:
                val_lr = lr_metrics[key]
                val_hr = hr_metrics[key]
                gains[f'{key}_gain'] = val_hr - val_lr
                ratios[f'{key}_ratio'] = val_hr / (val_lr + epsilon)
            
            # Composite Detail Gain Score (weighted combination)
            # Positive = HR has more detail than LR
            detail_gain_score = (
                0.25 * gains['laplacian_var_gain'] +
                0.20 * gains['high_freq_ratio_gain'] * 100 +  # Scale up small values
                0.25 * gains['gradient_mean_gain'] * 10 +
                0.15 * gains['edge_density_gain'] * 100 +
                0.15 * gains['local_contrast_gain'] * 10
            )
            
            # Quality Ratio (should be > 1.0 for valid super-resolution)
            quality_ratio = np.mean([
                ratios['laplacian_var_ratio'],
                ratios['entropy_ratio'],
                ratios['gradient_mean_ratio']
            ])
            
            # Detect suspicious patterns
            is_blurry = quality_ratio < 0.95  # HR worse than LR
            is_noisy = (ratios['laplacian_var_ratio'] > 3.0 and 
                       gains['gradient_std_gain'] < 0)  # High variance but disorganized
            is_duplicate = abs(quality_ratio - 1.0) < 0.01 and abs(detail_gain_score) < 0.1
            
            # Status determination
            if is_blurry:
                status = 'CRITICAL'
                issue_type = 'BLURRY_HR'
            elif is_duplicate:
                status = 'CRITICAL'
                issue_type = 'DUPLICATE'
            elif is_noisy:
                status = 'WARNING'
                issue_type = 'NOISY_ARTIFACTS'
            elif quality_ratio < 1.0:
                status = 'WARNING'
                issue_type = 'DETAIL_LOSS'
            elif quality_ratio > 2.0:
                status = 'WARNING'
                issue_type = 'SUSPICIOUS_HIGH'
            else:
                status = 'OK'
                issue_type = 'NORMAL'
            
            return {
                'filename': Path(lr_path).name,
                'group': extract_group_prefix(Path(lr_path).name),
                **{f'lr_{k}': float(v) for k, v in lr_metrics.items()},
                **{f'hr_{k}': float(v) for k, v in hr_metrics.items()},
                **{k: float(v) for k, v in gains.items()},
                **{k: float(v) for k, v in ratios.items()},
                'detail_gain_score': float(detail_gain_score),
                'quality_ratio': float(quality_ratio),
                'status': status,
                'issue_type': issue_type
            }
            
    except Exception as e:
        return {
            'filename': Path(lr_path).name, 
            'error': str(e), 
            'status': 'ERROR',
            'issue_type': 'PROCESSING_ERROR'
        }


def calculate_group_statistics(results, group_prefix_parts=2):
    """Calculate statistics for each group."""
    groups = defaultdict(list)
    
    for r in results:
        if 'error' not in r:
            group = extract_group_prefix(r['filename'], group_prefix_parts)
            groups[group].append(r)
    
    group_stats = {}
    for group_name, items in groups.items():
        quality_ratios = [i['quality_ratio'] for i in items]
        gain_scores = [i['detail_gain_score'] for i in items]
        
        critical = sum(1 for i in items if i['status'] == 'CRITICAL')
        warning = sum(1 for i in items if i['status'] == 'WARNING')
        ok = sum(1 for i in items if i['status'] == 'OK')
        
        # Detect specific issues
        blurry_count = sum(1 for i in items if i['issue_type'] == 'BLURRY_HR')
        dup_count = sum(1 for i in items if i['issue_type'] == 'DUPLICATE')
        
        group_stats[group_name] = {
            'count': len(items),
            'critical': critical,
            'warning': warning,
            'ok': ok,
            'blurry_count': blurry_count,
            'duplicate_count': dup_count,
            'mean_quality_ratio': float(np.mean(quality_ratios)),
            'std_quality_ratio': float(np.std(quality_ratios)),
            'mean_gain_score': float(np.mean(gain_scores)),
            'min_quality_ratio': float(np.min(quality_ratios)),
            'max_quality_ratio': float(np.max(quality_ratios)),
            'detail_retention': 100 * (ok / len(items)) if len(items) > 0 else 0,
            'items': items,
            'health_score': 100 * (ok + 0.5 * warning) / len(items)
        }
    
    return group_stats


def generate_executive_summary(total_results, group_stats, errors):
    """Generate narrative summary focused on detail quality."""
    total = len(total_results)
    critical = [r for r in total_results if r['status'] == 'CRITICAL']
    warnings = [r for r in total_results if r['status'] == 'WARNING']
    ok = [r for r in total_results if r['status'] == 'OK']
    
    quality_ratios = [r['quality_ratio'] for r in total_results]
    mean_ratio = np.mean(quality_ratios)
    std_ratio = np.std(quality_ratios)
    
    # Categorize issues
    blurry = [r for r in critical if r.get('issue_type') == 'BLURRY_HR']
    duplicates = [r for r in critical if r.get('issue_type') == 'DUPLICATE']
    
    # Determine grade
    if len(critical) == 0 and mean_ratio > 1.1:
        grade = "EXCELLENT"
        verdict = "Dataset exhibits strong detail preservation with consistent high-frequency content in HR pairs."
    elif len(blurry) == 0 and mean_ratio > 1.0:
        grade = "GOOD"
        verdict = "Dataset shows acceptable detail gain with minor variations in reconstruction quality."
    elif len(blurry) < total * 0.05:
        grade = "FAIR"
        verdict = "Dataset has minor issues with some blurry HR images, but generally preserves detail."
    else:
        grade = "POOR"
        verdict = "Dataset exhibits significant quality degradation. HR images do not contain expected detail gain."
    
    summary = f"""
EXECUTIVE SUMMARY - DETAIL GAIN VALIDATION
{'='*70}

VALIDATION GRADE: {grade}
OVERALL STATUS: {'PASS' if grade in ['EXCELLENT', 'GOOD'] else 'CONDITIONAL' if grade == 'FAIR' else 'FAIL'}

DATASET OVERVIEW:
  • Total Pairs Analyzed: {total:,}
  • Sequence Groups: {len(group_stats)}
  • Processing Errors: {len(errors)}

QUALITY METRICS:
  • Mean Quality Ratio: {mean_ratio:.3f} (target: >1.0)
  • Ratio Std Dev: {std_ratio:.3f} (lower is more consistent)
  • Detail Retention Rate: {100*len(ok)/total:.1f}%

ISSUE BREAKDOWN:
  ✅ Valid Detail Gain: {len(ok):,} ({100*len(ok)/total:.1f}%)
  ⚠️ Suspicious Patterns: {len(warnings):,} ({100*len(warnings)/total:.1f}%)
  ❌ Critical Failures: {len(critical):,} ({100*len(critical)/total:.1f}%)
     ├─ Blurry HR (worse than LR): {len(blurry)}
     └─ Likely Duplicates: {len(duplicates)}

DETECTION ANALYSIS:
"""
    
    if blurry:
        worst_blur = min(blurry, key=lambda x: x['quality_ratio'])
        summary += f"  🔍 Blurriest sample: {worst_blur['filename']} (ratio: {worst_blur['quality_ratio']:.3f})\n"
    
    if duplicates:
        summary += f"  🔄 Found {len(duplicates)} potential duplicate pairs (LR/HR identical)\n"
    
    summary += f"\nASSESSMENT:\n{verdict}\n"
    
    if mean_ratio < 1.0:
        summary += f"\n  ⚠️  WARNING: HR images contain LESS detail than LR on average (ratio {mean_ratio:.3f}).\n"
        summary += "      This suggests upscaling without reconstruction or processing errors.\n"
    elif mean_ratio > 1.5:
        summary += f"\n  ✓ HR images show {mean_ratio:.1f}x average detail improvement over LR.\n"
    
    if group_stats:
        worst_group = min(group_stats.items(), key=lambda x: x[1]['mean_quality_ratio'])
        best_group = max(group_stats.items(), key=lambda x: x[1]['mean_quality_ratio'])
        summary += f"\nGROUP INSIGHTS:\n"
        summary += f"  • Best Quality: {best_group[0]} (ratio: {best_group[1]['mean_quality_ratio']:.3f})\n"
        summary += f"  • Needs Review: {worst_group[0]} (ratio: {worst_group[1]['mean_quality_ratio']:.3f})\n"
    
    return summary


def generate_group_section(group_name, stats):
    """Generate detailed section for a group."""
    status_symbol = "🔴" if stats['critical'] > 0 else ("🟡" if stats['warning'] > 0 else "🟢")
    
    section = f"""
GROUP: {group_name} {status_symbol}
{'-'*70}
Population: {stats['count']} frames | Health: {stats['health_score']:.1f}%
Quality Ratio: {stats['mean_quality_ratio']:.3f} ± {stats['std_quality_ratio']:.3f}
Issues: {stats['ok']} OK | {stats['warning']} Warning | {stats['critical']} Critical
       ({stats['blurry_count']} blurry, {stats['duplicate_count']} duplicates)

Distribution:
  Min Ratio: {stats['min_quality_ratio']:.3f} | Max: {stats['max_quality_ratio']:.3f}
  Mean Gain Score: {stats['mean_gain_score']:.2f}

Worst Offenders:
"""
    
    sorted_items = sorted(stats['items'], key=lambda x: x['quality_ratio'])[:5]
    for i, item in enumerate(sorted_items, 1):
        symbol = "🔴" if item['status'] == 'CRITICAL' else "🟡"
        section += f"  {i}. {item['filename'][:45]:<46} Ratio: {item['quality_ratio']:.3f} {symbol} {item['issue_type']}\n"
    
    return section


def validate_detail_gain(lr_dir, hr_dir, num_workers=None, max_size=None, 
                         save_report=True, group_prefix_parts=2):
    """Main validation routine for detail gain."""
    lr_path = Path(lr_dir)
    hr_path = Path(hr_dir)
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    # Find pairs
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    lr_images = []
    for ext in image_extensions:
        lr_images.extend(lr_path.glob(ext))
        lr_images.extend(lr_path.glob(ext.upper()))
    
    lr_images = sorted(set(lr_images))
    
    tasks = []
    for lr_img_path in lr_images:
        hr_img_path = hr_path / lr_img_path.name
        if hr_img_path.exists():
            tasks.append((str(lr_img_path), str(hr_img_path), max_size))
    
    print(f"Found {len(tasks)} valid pairs")
    print(f"Processing with {num_workers} threads...")
    print("Metrics: Laplacian variance, FFT high-freq, Gradient density, Entropy\n")
    
    # Process
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(analyze_detail_gain, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Analyzing Detail") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result(timeout=30)
                    if 'error' not in result:
                        results.append(result)
                    else:
                        errors.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    errors.append({
                        'filename': Path(task[0]).name, 
                        'error': str(e), 
                        'status': 'ERROR'
                    })
                pbar.update(1)
    
    if not results:
        print(f"❌ No valid results! All {len(errors)} tasks failed.")
        return False
    
    # Analysis
    group_stats = calculate_group_statistics(results, group_prefix_parts)
    sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['mean_quality_ratio'])
    
    critical = [r for r in results if r['status'] == 'CRITICAL']
    warnings = [r for r in results if r['status'] == 'WARNING']
    ok = [r for r in results if r['status'] == 'OK']
    
    # Console output
    print("\n" + "="*70)
    print("DETAIL GAIN VALIDATION SUMMARY")
    print("="*70)
    print(f"Total analyzed: {len(results)}")
    print(f"Groups found: {len(group_stats)}")
    print(f"  ✅ Valid Gain: {len(ok)} ({100*len(ok)/len(results):.1f}%)")
    print(f"  ⚠️  Suspicious: {len(warnings)} ({100*len(warnings)/len(results):.1f}%)")
    print(f"  ❌ Critical: {len(critical)} ({100*len(critical)/len(results):.1f}%)")
    
    mean_ratio = np.mean([r['quality_ratio'] for r in results])
    print(f"\nMean Quality Ratio: {mean_ratio:.3f}")
    print(f"(HR detail / LR detail, should be > 1.0)")
    
    if critical:
        blurry = [r for r in critical if r.get('issue_type') == 'BLURRY_HR']
        dups = [r for r in critical if r.get('issue_type') == 'DUPLICATE']
        if blurry:
            print(f"\n⚠️  Found {len(blurry)} blurry HR images (less detail than LR)")
        if dups:
            print(f"⚠️  Found {len(dups)} potential duplicates")
    
    # Group table
    print(f"\n{'='*70}")
    print("GROUP RANKING (by quality ratio)")
    print(f"{'Group':<30} {'Count':<8} {'Avg Ratio':<12} {'Critical':<10} {'Status':<12}")
    print("-" * 70)
    
    for group_name, stats in sorted_groups[:10]:
        status = "🔴 FAIL" if stats['critical'] > 0 else ("🟡 CHECK" if stats['warning'] > 0 else "🟢 PASS")
        print(f"{group_name:<30} {stats['count']:<8} {stats['mean_quality_ratio']:.3f}       "
              f"{stats['critical']:<10} {status}")
    
    # Report generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"detail_gain_report_{timestamp}.txt"
    csv_path = f"detail_gain_report_{timestamp}.csv"
    
    if save_report and results:
        with open(report_path, 'w') as f:
            f.write("DETAIL GAIN VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LR Directory: {lr_dir}\n")
            f.write(f"HR Directory: {hr_dir}\n\n")
            
            f.write(generate_executive_summary(results, group_stats, errors))
            f.write("\n\n")
            
            # Global statistics table
            f.write("="*70 + "\n")
            f.write("GLOBAL STATISTICS\n")
            f.write("="*70 + "\n")
            f.write(f"{'Metric':<30} {'LR Mean':<12} {'HR Mean':<12} {'Gain':<12}\n")
            f.write("-" * 70 + "\n")
            
            metrics_keys = ['laplacian_var', 'high_freq_ratio', 'gradient_mean', 
                          'edge_density', 'entropy', 'local_contrast']
            
            for key in metrics_keys:
                lr_vals = [r[f'lr_{key}'] for r in results if f'lr_{key}' in r]
                hr_vals = [r[f'hr_{key}'] for r in results if f'hr_{key}' in r]
                if lr_vals and hr_vals:
                    lr_mean = np.mean(lr_vals)
                    hr_mean = np.mean(hr_vals)
                    gain = hr_mean - lr_mean
                    f.write(f"{key:<30} {lr_mean:<12.4f} {hr_mean:<12.4f} {gain:+12.4f}\n")
            
            f.write("\n\n")
            
            # Group details
            f.write("="*70 + "\n")
            f.write("DETAILED GROUP ANALYSIS\n")
            f.write("="*70 + "\n")
            
            for group_name, stats in sorted_groups:
                f.write(generate_group_section(group_name, stats))
                f.write("\n")
            
            # Final summary
            f.write("\n" + "="*70 + "\n")
            f.write("VALIDATION CONCLUSION\n")
            f.write("="*70 + "\n\n")
            
            if mean_ratio >= 1.1 and len(critical) == 0:
                f.write("✅ PASSED: HR images consistently contain more detail than LR sources.\n")
                f.write("   The super-resolution pipeline is adding genuine high-frequency content.\n")
            elif mean_ratio >= 1.0 and len(critical) < len(results) * 0.05:
                f.write("⚠️  CONDITIONAL: Minor issues detected but generally acceptable detail gain.\n")
                f.write("   Review flagged sequences before production use.\n")
            else:
                f.write("❌ FAILED: Significant detail preservation issues detected.\n")
                f.write(f"   Mean quality ratio ({mean_ratio:.3f}) indicates ")
                if mean_ratio < 1.0:
                    f.write("HR images are blurrier than LR sources.\n")
                else:
                    f.write("inconsistent reconstruction quality.\n")
                f.write("   RECOMMENDATION: Do not use for training until corrected.\n")
            
            f.write(f"\nAnalysis based on {len(results)} image pairs across {len(group_stats)} groups.\n")
        
        # CSV with all metrics
        with open(csv_path, 'w', newline='') as f:
            if results:
                # Get all fieldnames from first result
                sample = results[0]
                fieldnames = ['rank', 'group', 'filename', 'quality_ratio', 'detail_gain_score',
                            'status', 'issue_type']
                # Add all metric fields
                for key in sample.keys():
                    if key not in fieldnames and key != 'error':
                        fieldnames.append(key)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Sort by quality ratio (worst first)
                sorted_results = sorted(results, key=lambda x: x['quality_ratio'])
                
                for i, r in enumerate(sorted_results, 1):
                    row = {'rank': i, **{k: v for k, v in r.items() if k != 'error'}}
                    writer.writerow(row)
        
        print(f"\n📄 Reports saved:")
        print(f"   • {report_path}")
        print(f"   • {csv_path}")
    
    # Return verdict
    if mean_ratio >= 1.0 and len(critical) < len(results) * 0.05:
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Detail gain validation (detects blurry SR)')
    parser.add_argument('--lr_dir', type=str, default='./data/lr_frames')
    parser.add_argument('--hr_dir', type=str, default='./data/hr_frames')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=1024, 
                       help='Max dimension for processing speed (default: 1024)')
    parser.add_argument('--group_parts', type=int, default=2,
                       help='Number of underscore segments for grouping')
    parser.add_argument('--no_save', action='store_true')
    
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 8)
    
    success = validate_detail_gain(
        args.lr_dir, args.hr_dir, 
        args.workers, args.max_size,
        save_report=not args.no_save,
        group_prefix_parts=args.group_parts
    )
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
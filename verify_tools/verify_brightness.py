#!/usr/bin/env python3
"""
Multi-core brightness validation with detailed grouping and executive reporting
Groups images by prefix (e.g., clip01_sequence00) with statistical analysis
"""

import os
import csv
import re
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
    RESAMPLE_FILTER = Image.LANCZOS  # Pillow < 9.1.0


def extract_group_prefix(filename, parts=2):
    """
    Extract group prefix from filename.
    e.g., 'clip01_sequence00_frame001.png' -> 'clip01_sequence00'
    """
    stem = Path(filename).stem
    segments = stem.split('_')
    if len(segments) >= parts:
        return '_'.join(segments[:parts])
    return stem  # Fallback if not enough segments


def analyze_brightness(args):
    """Worker function."""
    lr_path, hr_path, max_size = args
    
    try:
        if not os.path.exists(lr_path) or not os.path.exists(hr_path):
            return {'filename': Path(lr_path).name, 'error': 'File not found', 'status': 'ERROR'}
        
        with Image.open(lr_path) as lr_img, Image.open(hr_path) as hr_img:
            lr_img = lr_img.convert('RGB')
            hr_img = hr_img.convert('RGB')
            
            if max_size and max_size > 0:
                lr_img.thumbnail((max_size, max_size), RESAMPLE_FILTER)
                hr_img.thumbnail((max_size, max_size), RESAMPLE_FILTER)
            
            lr_arr = np.array(lr_img).astype(np.float32) / 255.0
            hr_arr = np.array(hr_img).astype(np.float32) / 255.0
            
            lr_mean = np.mean(lr_arr)
            hr_mean = np.mean(hr_arr)
            
            if lr_mean < 0.001:
                return {'filename': Path(lr_path).name, 'error': 'Near-black LR image', 'status': 'ERROR'}
            
            lr_channel_means = np.mean(lr_arr, axis=(0, 1))
            hr_channel_means = np.mean(hr_arr, axis=(0, 1))
            
            mean_diff = hr_mean - lr_mean
            channel_diffs = hr_channel_means - lr_channel_means
            ratio = hr_mean / lr_mean
            
            abs_diff = abs(mean_diff)
            ratio_error = abs(ratio - 1.0)
            severity = abs_diff + (ratio_error * 0.5)
            
            return {
                'filename': Path(lr_path).name,
                'lr_mean': float(lr_mean),
                'hr_mean': float(hr_mean),
                'mean_diff': float(mean_diff),
                'abs_diff': float(abs_diff),
                'brightness_ratio': float(ratio),
                'ratio_error': float(ratio_error),
                'severity': float(severity),
                'r_diff': float(channel_diffs[0]),
                'g_diff': float(channel_diffs[1]),
                'b_diff': float(channel_diffs[2]),
                'status': 'CRITICAL' if abs_diff > 0.05 else ('WARNING' if abs_diff > 0.02 else 'OK')
            }
    except Exception as e:
        return {'filename': Path(lr_path).name, 'error': str(e), 'status': 'ERROR'}


def calculate_group_statistics(results, group_prefix_parts=2):
    """Calculate statistics for each group and overall."""
    groups = defaultdict(list)
    
    for r in results:
        group = extract_group_prefix(r['filename'], group_prefix_parts)
        groups[group].append(r)
    
    group_stats = {}
    for group_name, items in groups.items():
        diffs = [i['mean_diff'] for i in items]
        abs_diffs = [i['abs_diff'] for i in items]
        severities = [i['severity'] for i in items]
        
        critical = sum(1 for i in items if i['status'] == 'CRITICAL')
        warning = sum(1 for i in items if i['status'] == 'WARNING')
        ok = sum(1 for i in items if i['status'] == 'OK')
        
        group_stats[group_name] = {
            'count': len(items),
            'critical': critical,
            'warning': warning,
            'ok': ok,
            'critical_pct': 100 * critical / len(items),
            'warning_pct': 100 * warning / len(items),
            'ok_pct': 100 * ok / len(items),
            'mean_diff': float(np.mean(diffs)),
            'std_diff': float(np.std(diffs)),
            'mean_abs_diff': float(np.mean(abs_diffs)),
            'max_diff': float(np.max(abs_diffs)),
            'mean_severity': float(np.mean(severities)),
            'items': items,
            'health_score': 100 * (ok + 0.5 * warning) / len(items)  # Weighted score
        }
    
    return group_stats, groups


def generate_executive_summary(total_results, group_stats, errors):
    """Generate a narrative summary of findings."""
    total = len(total_results)
    critical = sum(1 for r in total_results if r['status'] == 'CRITICAL')
    warning = sum(1 for r in total_results if r['status'] == 'WARNING')
    ok = sum(1 for r in total_results if r['status'] == 'OK')
    
    mean_diff = np.mean([r['mean_diff'] for r in total_results])
    mean_abs = np.mean([r['abs_diff'] for r in total_results])
    
    # Determine overall grade
    if critical == 0 and mean_abs < 0.01:
        grade = "EXCELLENT"
        verdict = "Dataset exhibits exceptional brightness consistency with minimal deviation between LR and HR pairs."
    elif critical < total * 0.05 and mean_abs < 0.03:
        grade = "GOOD"
        verdict = "Dataset shows good brightness matching with minor acceptable variations."
    elif critical < total * 0.15 and mean_abs < 0.05:
        grade = "FAIR"
        verdict = "Dataset has moderate brightness inconsistencies that may require attention."
    else:
        grade = "POOR"
        verdict = "Dataset exhibits significant brightness mismatches requiring immediate correction."
    
    # Find worst group
    worst_group = max(group_stats.items(), key=lambda x: x[1]['mean_abs_diff']) if group_stats else None
    
    summary = f"""
EXECUTIVE SUMMARY
{'='*70}

VALIDATION GRADE: {grade}
OVERALL STATUS: {'PASS' if grade in ['EXCELLENT', 'GOOD'] else 'CONDITIONAL' if grade == 'FAIR' else 'FAIL'}

DATASET OVERVIEW:
  • Total Images Analyzed: {total:,}
  • Sequence Groups Identified: {len(group_stats)}
  • Processing Errors: {len(errors)}

DISTRIBUTION ANALYSIS:
  • Compliant (OK): {ok:,} images ({100*ok/total:.1f}%)
  • Minor Deviation (Warning): {warning:,} images ({100*warning/total:.1f}%)  
  • Critical Mismatch: {critical:,} images ({100*critical/total:.1f}%)

STATISTICAL METRICS:
  • Mean Brightness Offset: {mean_diff:+.4f} ({mean_diff*100:+.2f}%)
  • Mean Absolute Deviation: {mean_abs:.4f}
  • Standard Deviation: {np.std([r['mean_diff'] for r in total_results]):.4f}

GROUP ANALYSIS:
  • Best Performing Group: {min(group_stats.items(), key=lambda x: x[1]['mean_abs_diff'])[0] if group_stats else 'N/A'}
  • Most Problematic Group: {worst_group[0] if worst_group else 'N/A'} (Avg Dev: {worst_group[1]['mean_abs_diff']:.4f})
  • Group Consistency: {np.mean([g['std_diff'] for g in group_stats.values()]):.4f} (lower is better)

ASSESSMENT:
{verdict}
"""
    
    if worst_group and worst_group[1]['critical'] > 0:
        summary += f"\n  ⚠️  Attention required for group '{worst_group[0]}' with {worst_group[1]['critical']} critical issues.\n"
    
    if mean_diff > 0.01:
        summary += f"\n  📊 Note: HR images are systematically brighter than LR by ~{mean_diff*100:.1f}% on average.\n"
    elif mean_diff < -0.01:
        summary += f"\n  📊 Note: HR images are systematically darker than LR by ~{abs(mean_diff)*100:.1f}% on average.\n"
    
    return summary


def generate_group_report_section(group_name, stats):
    """Generate detailed section for a single group."""
    status_symbol = "🔴" if stats['critical'] > 0 else ("🟡" if stats['warning'] > 0 else "🟢")
    overall_status = "CRITICAL" if stats['critical'] > 0 else ("WARNING" if stats['warning'] > 0 else "OK")
    
    section = f"""
GROUP: {group_name} {status_symbol}
{'-'*70}
Population: {stats['count']} frames | Health Score: {stats['health_score']:.1f}/100
Status Distribution: {stats['ok']} OK | {stats['warning']} Warning | {stats['critical']} Critical

Statistical Profile:
  Mean Difference: {stats['mean_diff']:+.4f} ± {stats['std_diff']:.4f}
  Absolute Deviation: {stats['mean_abs_diff']:.4f} (max: {stats['max_diff']:.4f})
  Mean Severity Index: {stats['mean_severity']:.4f}

Breakdown: {stats['critical_pct']:.1f}% Critical, {stats['warning_pct']:.1f}% Warning, {stats['ok_pct']:.1f}% OK

Top Issues (Worst 5):
"""
    
    # Sort by severity and show top 5
    sorted_items = sorted(stats['items'], key=lambda x: x['severity'], reverse=True)[:5]
    for i, item in enumerate(sorted_items, 1):
        symbol = "🔴" if item['status'] == 'CRITICAL' else "🟡"
        section += f"  {i}. {item['filename']:<45} Diff: {item['mean_diff']:+.4f}  {symbol}\n"
    
    return section


def validate_dataset(lr_dir, hr_dir, num_workers=None, max_size=None, 
                     save_report=True, show_top_n=50, group_prefix_parts=2):
    """Analyze all pairs with grouping and detailed reporting."""
    lr_path = Path(lr_dir)
    hr_path = Path(hr_dir)
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    # Find pairs
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
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
    print(f"Grouping by first {group_prefix_parts} underscore-separated segments\n")
    
    # Process with ThreadPoolExecutor
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(analyze_brightness, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Analyzing") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result(timeout=30)
                    if result and 'error' not in result:
                        results.append(result)
                    else:
                        errors.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    errors.append({'filename': Path(task[0]).name, 'error': str(e), 'status': 'ERROR'})
                pbar.update(1)
    
    if not results:
        print(f"❌ No valid results! All {len(errors)} tasks failed.")
        if errors:
            print("First few errors:")
            for e in errors[:3]:
                print(f"  {e.get('filename', 'unknown')}: {e.get('error', 'unknown error')}")
        return False
    
    # Group analysis
    group_stats, groups = calculate_group_statistics(results, group_prefix_parts)
    
    # Sort groups by severity (worst first)
    sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['mean_abs_diff'], reverse=True)
    
    # Global statistics
    mean_diffs = [r['mean_diff'] for r in results]
    critical = [r for r in results if r['status'] == 'CRITICAL']
    warnings = [r for r in results if r['status'] == 'WARNING']
    ok = [r for r in results if r['status'] == 'OK']
    
    # Console output
    print("\n" + "="*70)
    print("BRIGHTNESS VALIDATION SUMMARY")
    print("="*70)
    print(f"Total analyzed: {len(results)}")
    print(f"Groups found: {len(group_stats)}")
    print(f"  ✅ OK: {len(ok)} ({100*len(ok)/len(results):.1f}%)")
    print(f"  ⚠️  WARNING: {len(warnings)} ({100*len(warnings)/len(results):.1f}%)")
    print(f"  ❌ CRITICAL: {len(critical)} ({100*len(critical)/len(results):.1f}%)")
    if errors:
        print(f"  💥 Errors: {len(errors)}")
    
    print(f"\nMean difference: {np.mean(mean_diffs):+.4f}")
    print(f"Mean ratio: {np.mean([r['brightness_ratio'] for r in results]):.4f}")
    
    # Group summary in console
    print(f"\n{'='*70}")
    print("GROUP RANKING (by average deviation)")
    print(f"{'='*70}")
    print(f"{'Group':<30} {'Count':<8} {'Avg Diff':<10} {'Critical':<10} {'Status':<10}")
    print("-" * 70)
    
    for group_name, stats in sorted_groups[:10]:  # Show top 10 groups
        status = "🔴 CRITICAL" if stats['critical'] > 0 else ("🟡 WARNING" if stats['warning'] > 0 else "🟢 OK")
        print(f"{group_name:<30} {stats['count']:<8} {stats['mean_diff']:+.4f}    {stats['critical']:<10} {status}")
    
    # Detailed report generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"brightness_report_{timestamp}.txt"
    csv_path = f"brightness_report_{timestamp}.csv"
    
    if save_report and results:
        with open(report_path, 'w') as f:
            # Header
            f.write("BRIGHTNESS VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LR Directory: {lr_dir}\n")
            f.write(f"HR Directory: {hr_dir}\n")
            f.write(f"Grouping Method: First {group_prefix_parts} underscore segments\n\n")
            
            # Executive Summary
            f.write(generate_executive_summary(results, group_stats, errors))
            f.write("\n\n")
            
            # Group Summary Table
            f.write("="*70 + "\n")
            f.write("GROUP ANALYSIS RANKING\n")
            f.write("="*70 + "\n")
            f.write(f"{'Rank':<6} {'Group':<30} {'Frames':<8} {'Avg Diff':<10} {'Max Diff':<10} {'Health':<8}\n")
            f.write("-" * 80 + "\n")
            
            for i, (group_name, stats) in enumerate(sorted_groups, 1):
                f.write(f"{i:<6} {group_name:<30} {stats['count']:<8} "
                       f"{stats['mean_diff']:+.4f}    {stats['max_diff']:.4f}    "
                       f"{stats['health_score']:.1f}%\n")
            
            f.write("\n\n")
            
            # Detailed Group Sections
            f.write("="*70 + "\n")
            f.write("DETAILED FINDINGS BY GROUP\n")
            f.write("="*70 + "\n")
            
            for group_name, stats in sorted_groups:
                f.write(generate_group_report_section(group_name, stats))
                f.write("\n")
            
            # Final Summary Statement
            f.write("\n" + "="*70 + "\n")
            f.write("FINAL SUMMARY STATEMENT\n")
            f.write("="*70 + "\n\n")
            
            total_critical = len(critical)
            total_warning = len(warnings)
            
            if total_critical == 0 and np.mean(np.abs(mean_diffs)) < 0.01:
                f.write("✅ VALIDATION PASSED: The dataset demonstrates excellent brightness consistency ")
                f.write("across all sequence groups. No corrective action required.\n\n")
                f.write("RECOMMENDATION: Dataset is approved for training/use.\n")
            elif total_critical < len(results) * 0.05:
                f.write("⚠️  VALIDATION CONDITIONAL: Minor brightness variations detected. ")
                f.write(f"Only {total_critical} critical issues found across {len(results)} images. ")
                f.write("Dataset is usable but monitoring recommended.\n\n")
                f.write("RECOMMENDATION: Review flagged sequences before production use.\n")
            else:
                f.write("❌ VALIDATION FAILED: Significant brightness mismatches detected. ")
                f.write(f"{total_critical} images ({100*total_critical/len(results):.1f}%) exceed critical thresholds. ")
                f.write("This indicates systematic processing errors or source data issues.\n\n")
                f.write("RECOMMENDATION: Do not use for training until corrected. ")
                if group_stats:
                    worst = max(group_stats.items(), key=lambda x: x[1]['critical'])
                    f.write(f"Priority: Investigate group '{worst[0]}' ({worst[1]['critical']} critical failures).\n")
            
            f.write(f"\nReport generated with {len(results)} valid samples and {len(errors)} errors.\n")
        
        # Enhanced CSV with group column
        with open(csv_path, 'w', newline='') as f:
            if results:
                fieldnames = ['rank', 'group', 'filename', 'lr_mean', 'hr_mean', 
                            'mean_diff', 'abs_diff', 'brightness_ratio', 
                            'ratio_error', 'severity', 'r_diff', 'g_diff', 'b_diff', 
                            'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Sort by group then by severity within group
                results_sorted = sorted(results, 
                    key=lambda x: (extract_group_prefix(x['filename'], group_prefix_parts), -x['severity']))
                
                for i, r in enumerate(results_sorted, 1):
                    group = extract_group_prefix(r['filename'], group_prefix_parts)
                    row = {
                        'rank': i,
                        'group': group,
                        **{k: v for k, v in r.items() if k != 'error'}
                    }
                    writer.writerow(row)
        
        print(f"\n📄 Detailed reports saved:")
        print(f"   • {report_path}")
        print(f"   • {csv_path}")
    
    # Final verdict
    mean_abs = np.mean(np.abs(mean_diffs))
    if mean_abs < 0.01 and len(critical) == 0:
        print("\n✅ PASSED: Excellent brightness matching")
        return True
    elif mean_abs < 0.03 and len(critical) < len(results) * 0.05:
        print("\n⚠️  WARNING: Minor brightness issues")
        return True
    else:
        print("\n❌ FAILED: Significant brightness mismatch!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Brightness validation with group analysis')
    parser.add_argument('--lr_dir', type=str, default='./data/lr_frames')
    parser.add_argument('--hr_dir', type=str, default='./data/hr_frames')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=None)
    parser.add_argument('--show_top', type=int, default=50)
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--group_parts', type=int, default=2,
                       help='Number of underscore-separated parts to use as group key (default: 2)')
    
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 8)
    
    is_valid = validate_dataset(
        args.lr_dir, args.hr_dir, 
        args.workers, args.max_size,
        save_report=not args.no_save,
        show_top_n=args.show_top,
        group_prefix_parts=args.group_parts
    )
    
    exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
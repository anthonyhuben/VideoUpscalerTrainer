#!/usr/bin/env python3
"""
Export SuperUltraCompact model to NCNN format
Pipeline: PyTorch -> ONNX -> NCNN
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import shutil

# Import your model architecture
from train_upscaler import SuperUltraCompact, load_pretrained


def export_to_onnx(model, output_path, input_size=(1, 3, 540, 960)):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    print(f"Exporting ONNX to {output_path}...")
    print(f"Input shape: {input_size}")
    
    try:
        # Try newer API first (PyTorch 2.0+)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
    except Exception as e:
        print(f"Standard export failed: {e}")
        print("Trying dynamo=False fallback...")
        # Fallback for older/newer compatibility
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            opset_version=11,
            dynamo=False  # Disable new exporter
        )
    
    print(f"✅ ONNX model saved: {output_path}")
    
    # Try to simplify if onnxsim is available (optional)
    try:
        import onnx
        from onnxsim import simplify
        
        print("Simplifying ONNX model...")
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print("✅ ONNX model simplified")
        else:
            print("⚠️  Simplification check failed, using original")
    except ImportError:
        print("ℹ️  onnxsim not available (optional), skipping simplification")
    except Exception as e:
        print(f"⚠️  Simplification failed: {e}")


def check_ncnn_tools():
    """Check if NCNN tools are available"""
    onnx2ncnn = shutil.which('onnx2ncnn')
    ncnnoptimize = shutil.which('ncnnoptimize')
    
    if not onnx2ncnn:
        print("❌ onnx2ncnn not found in PATH")
        print("   Install NCNN: https://github.com/Tencent/ncnn")
        print("   MacOS: brew install ncnn")
        return False, False
    
    return True, ncnnoptimize is not None


def convert_onnx_to_ncnn(onnx_path, output_dir):
    """Convert ONNX to NCNN using onnx2ncnn"""
    import subprocess
    
    param_path = os.path.join(output_dir, "model.param")
    bin_path = os.path.join(output_dir, "model.bin")
    
    try:
        result = subprocess.run(
            ['onnx2ncnn', onnx_path, param_path, bin_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ NCNN model saved:")
        print(f"   Param: {param_path}")
        print(f"   Bin:   {bin_path}")
        return param_path, bin_path
    except subprocess.CalledProcessError as e:
        print(f"❌ onnx2ncnn failed: {e}")
        print(f"   stderr: {e.stderr}")
        return None, None


def optimize_ncnn(param_path, bin_path):
    """Optimize NCNN model for inference"""
    import subprocess
    
    try:
        result = subprocess.run(
            ['ncnnoptimize', param_path, bin_path, param_path, bin_path, '0'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ NCNN model optimized")
        return True
    except Exception as e:
        print(f"⚠️  Optimization failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export to NCNN format')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Input .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default='./ncnn_model',
                       help='Output directory')
    parser.add_argument('--input_size', type=int, nargs=4, 
                       default=[1, 3, 540, 960],
                       help='Input shape (N C H W)')
    parser.add_argument('--onnx_only', action='store_true',
                       help='Export ONNX only, skip NCNN conversion')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check NCNN availability early
    has_ncnn, has_optimize = check_ncnn_tools()
    if not args.onnx_only and not has_ncnn:
        print("\n⚠️  NCNN tools not found. Will export ONNX only.")
        print("   To install NCNN: brew install ncnn (macOS)")
        args.onnx_only = True
    
    # Setup device
    device = torch.device('cpu')
    
    # Load model
    print("Loading model...")
    try:
        model = SuperUltraCompact(in_nc=3, out_nc=3, nf=24, nc=8, scale=2)
        model = load_pretrained(model, args.checkpoint, device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    input_size = tuple(args.input_size)
    
    try:
        export_to_onnx(model, onnx_path, input_size)
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        sys.exit(1)
    
    # Convert to NCNN if requested and available
    if not args.onnx_only:
        param_path, bin_path = convert_onnx_to_ncnn(onnx_path, args.output_dir)
        
        if param_path and has_optimize:
            optimize_ncnn(param_path, bin_path)
        
        # Optionally remove intermediate ONNX
        # os.remove(onnx_path)
    
    print(f"\n✅ Export complete!")
    if args.onnx_only:
        print(f"   ONNX only: {onnx_path}")
        print("   Use this ONNX with onnx2ncnn manually or online converters")
    else:
        print(f"   Files in: {args.output_dir}")


if __name__ == '__main__':
    main()
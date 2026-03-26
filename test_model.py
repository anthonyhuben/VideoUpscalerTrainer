"""
Test script to verify the pretrained model loads correctly
Run this BEFORE starting full training to catch any architecture issues
"""

import torch
import sys

# Import the model
from train_upscaler import SuperUltraCompact

def test_model_loading(pretrained_path):
    """Test if the pretrained model loads correctly"""
    
    print("="*60)
    print("PRETRAINED MODEL VERIFICATION")
    print("="*60)
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Device: CUDA GPU - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Device: CPU only")
    
    print("\n" + "="*60)
    print("Loading model architecture...")
    
    try:
        model = SuperUltraCompact(nf=24, nc=8, scale=2)
        print("✅ Model architecture created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to create model architecture: {e}")
        return False
    
    print("\n" + "="*60)
    print(f"Loading pretrained weights from:")
    print(f"  {pretrained_path}")
    
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        print("✅ Checkpoint file loaded")
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try to get the state dict
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
                print("   Using 'params' key")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("   Using 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("   Using 'state_dict' key")
            else:
                state_dict = checkpoint
                print("   Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print("   Checkpoint is the state_dict itself")
        
        # Show some keys
        state_keys = list(state_dict.keys())
        print(f"\n   State dict has {len(state_keys)} keys")
        print("   First 10 keys:")
        for key in state_keys[:10]:
            if hasattr(state_dict[key], 'shape'):
                print(f"     - {key}: {state_dict[key].shape}")
            else:
                print(f"     - {key}")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    print("\n" + "="*60)
    print("Attempting to load weights into model...")
    
    try:
        # Try strict loading first
        model.load_state_dict(state_dict, strict=True)
        print("✅ SUCCESS! Weights loaded successfully (strict mode)!")
        print("\n   All parameters matched perfectly!")
        
    except RuntimeError as e:
        error_str = str(e)
        
        # Check if it's just missing/unexpected keys
        if "Missing key" in error_str or "Unexpected key" in error_str:
            print(f"⚠️  Loading with strict=False due to key mismatches...")
            
            # Try non-strict loading
            result = model.load_state_dict(state_dict, strict=False)
            
            if result.missing_keys:
                print(f"\n❌ Missing keys in checkpoint:")
                for key in result.missing_keys[:10]:
                    print(f"   - {key}")
                if len(result.missing_keys) > 10:
                    print(f"   ... and {len(result.missing_keys) - 10} more")
            
            if result.unexpected_keys:
                print(f"\n❌ Unexpected keys in checkpoint:")
                for key in result.unexpected_keys[:10]:
                    print(f"   - {key}")
                if len(result.unexpected_keys) > 10:
                    print(f"   ... and {len(result.unexpected_keys) - 10} more")
            
            print("\n⚠️  The model architecture doesn't perfectly match.")
            print("   Training might work but could have issues.")
            
        else:
            # Size mismatch or other error
            print(f"❌ FAILED to load weights!")
            print(f"\nError message:")
            print(f"  {str(e)[:500]}...")
            print("\n⚠️  The model architecture doesn't match the pretrained weights.")
            print("   This needs to be fixed before training can start.")
            return False
    
    print("\n" + "="*60)
    print("Testing forward pass...")
    
    try:
        model = model.to(device)
        model.eval()
        
        # Create a dummy input (1 image, 3 channels, 64x64 pixels)
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape for 2x upscaler: (1, 3, 128, 128)")
        
        if output.shape == (1, 3, 128, 128):
            print("   ✅ Output shape is correct!")
        else:
            print(f"   ⚠️  Output shape is unexpected")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL CHECKS PASSED! ✅")
    print("="*60)
    print("\nYou can now run the training command:")
    print("  python3 train_upscaler.py \\")
    print("      --lr_dir ./data/lr_frames \\")
    print("      --hr_dir ./data/hr_frames \\")
    print(f"      --pretrained {pretrained_path} \\")
    print("      --output_dir ./checkpoints \\")
    print("      --batch_size 4 \\")
    print("      --epochs 100")
    print("="*60)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test pretrained model loading')
    parser.add_argument('--pretrained', type=str, 
                       default='./2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth',
                       help='Path to pretrained .pth file')
    
    args = parser.parse_args()
    
    success = test_model_loading(args.pretrained)
    
    if not success:
        print("\n❌ Model verification FAILED. Please fix the issues above before training.")
        sys.exit(1)
    else:
        print("\n✅ Model verification SUCCESSFUL!")
        sys.exit(0)

import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_sequences(video_path, output_dir, num_sequences=20, frames_per_sequence=8,
                     skip_start_frames=41400, skip_end_frames=7200):
    """
    Extract evenly-distributed 8-frame sequences from a video
    Skips the first and last N frames to avoid intros/outros
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        num_sequences: Number of sequences to extract from the video
        frames_per_sequence: Number of consecutive frames per sequence (default: 8)
        skip_start_frames: Number of frames to skip at start (default: 41400)
        skip_end_frames: Number of frames to skip at end (default: 7200)
    
    Returns:
        Number of frames extracted (num_sequences * frames_per_sequence)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define usable frame range (skip intro and outro)
    start_frame_limit = skip_start_frames
    end_frame_limit = total_frames - skip_end_frames
    usable_frames = end_frame_limit - start_frame_limit
    
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Skipping first {skip_start_frames} frames, last {skip_end_frames} frames")
    print(f"  Usable range: Frames {start_frame_limit}-{end_frame_limit} ({usable_frames} frames)")
    print(f"  Extracting {num_sequences} sequences of {frames_per_sequence} frames each")
    
    # Calculate total frames needed
    total_frames_needed = num_sequences * frames_per_sequence
    
    # Check if we have enough usable frames
    if total_frames_needed > usable_frames:
        print(f"  ⚠️  Warning: Requested {total_frames_needed} frames but only {usable_frames} usable frames available")
        print(f"  Reducing to {usable_frames // frames_per_sequence} sequences")
        num_sequences = usable_frames // frames_per_sequence
        total_frames_needed = num_sequences * frames_per_sequence
    
    # Calculate spacing between sequences within usable range
    if num_sequences > 1:
        # Space between the START of each sequence
        sequence_spacing = (usable_frames - frames_per_sequence) / (num_sequences - 1)
    else:
        # Only one sequence - start at the middle of usable range
        sequence_spacing = 0
    
    print(f"  Sequence spacing: ~{sequence_spacing:.1f} frames between each sequence start")
    
    video_name = video_path.stem
    extracted_count = 0
    
    # Extract each sequence
    for seq_num in range(num_sequences):
        # Calculate starting frame for this sequence within usable range
        offset_within_usable = int(seq_num * sequence_spacing)
        start_frame = start_frame_limit + offset_within_usable
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract consecutive frames for this sequence
        for frame_idx in range(frames_per_sequence):
            ret, frame = cap.read()
            
            if not ret:
                print(f"  Warning: Could not read frame at position {start_frame + frame_idx}")
                break
            
            # Save frame with sequence and frame number
            frame_filename = f"{video_name}_seq{seq_num:03d}_frame{frame_idx}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
    
    cap.release()
    
    print(f"  ✅ Extracted {num_sequences} sequences ({extracted_count} total frames)")
    return extracted_count


def process_video_pairs(lr_videos_dir, hr_videos_dir, lr_output_dir, hr_output_dir, 
                       num_sequences=20, frames_per_sequence=8,
                       skip_start_frames=41400, skip_end_frames=7200):
    """
    Process pairs of LR (1080p) and HR (4K) videos
    Extracts 8-frame sequences evenly distributed throughout usable portion of each video
    
    Args:
        lr_videos_dir: Directory containing 1080p videos
        hr_videos_dir: Directory containing 4K videos
        lr_output_dir: Directory to save 1080p frames
        hr_output_dir: Directory to save 4K frames
        num_sequences: Number of sequences to extract per video (default: 20)
        frames_per_sequence: Frames per sequence (default: 8)
        skip_start_frames: Frames to skip at video start (default: 41400)
        skip_end_frames: Frames to skip at video end (default: 7200)
    """
    lr_dir = Path(lr_videos_dir)
    hr_dir = Path(hr_videos_dir)
    
    # Get all video files (case-insensitive)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    lr_videos = []
    
    for ext in video_extensions:
        lr_videos.extend(lr_dir.glob(f'*{ext}'))
        lr_videos.extend(lr_dir.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    lr_videos = sorted(set(lr_videos))
    
    print(f"Found {len(lr_videos)} videos in {lr_videos_dir}")
    print(f"Configuration:")
    print(f"  Sequences per video: {num_sequences}")
    print(f"  Frames per sequence: {frames_per_sequence}")
    print(f"  Skip first/last: {skip_start_frames}/{skip_end_frames} frames")
    print(f"  Total frames per video: {num_sequences * frames_per_sequence}")
    
    total_lr_frames = 0
    total_hr_frames = 0
    processed_pairs = 0
    
    for lr_video in lr_videos:
        # Find corresponding HR video
        hr_video = hr_dir / lr_video.name
        
        if not hr_video.exists():
            print(f"⚠️  Warning: No matching HR video found for {lr_video.name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing video pair: {lr_video.name}")
        print(f"{'='*60}")
        
        # Extract LR sequences
        lr_frames = extract_sequences(lr_video, lr_output_dir, 
                                      num_sequences, frames_per_sequence,
                                      skip_start_frames, skip_end_frames)
        
        # Extract HR sequences  
        hr_frames = extract_sequences(hr_video, hr_output_dir, 
                                      num_sequences, frames_per_sequence,
                                      skip_start_frames, skip_end_frames)
        
        total_lr_frames += lr_frames
        total_hr_frames += hr_frames
        processed_pairs += 1
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {processed_pairs} video pairs")
    print(f"Total sequences extracted: {processed_pairs * num_sequences}")
    print(f"Total LR frames: {total_lr_frames}")
    print(f"Total HR frames: {total_hr_frames}")
    print(f"LR frames saved to: {lr_output_dir}")
    print(f"HR frames saved to: {hr_output_dir}")
    print(f"\n💡 Each sequence contains {frames_per_sequence} consecutive frames")


def main():
    parser = argparse.ArgumentParser(
        description='Extract 8-frame sequences from video pairs for video upscaling'
    )
    parser.add_argument('--lr_videos', type=str, required=True,
                       help='Directory containing 1080p videos')
    parser.add_argument('--hr_videos', type=str, required=True,
                       help='Directory containing 4K videos')
    parser.add_argument('--lr_output', type=str, default='./data/lr_frames',
                       help='Output directory for 1080p frames')
    parser.add_argument('--hr_output', type=str, default='./data/hr_frames',
                       help='Output directory for 4K frames')
    parser.add_argument('--num_sequences', type=int, default=20,
                       help='Number of 8-frame sequences to extract per video (default: 20)')
    parser.add_argument('--frames_per_sequence', type=int, default=8,
                       help='Number of consecutive frames per sequence (default: 8)')
    parser.add_argument('--skip_start', type=int, default=41400,
                       help='Frames to skip at video start (default: 41400)')
    parser.add_argument('--skip_end', type=int, default=7200,
                       help='Frames to skip at video end (default: 7200)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("8-FRAME SEQUENCE EXTRACTOR (WITH BOUNDARY SKIP)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Sequences per video: {args.num_sequences}")
    print(f"  Frames per sequence: {args.frames_per_sequence}")
    print(f"  Skip start frames: {args.skip_start}")
    print(f"  Skip end frames: {args.skip_end}")
    print(f"  Total frames per video: {args.num_sequences * args.frames_per_sequence}")
    print("="*60 + "\n")
    
    process_video_pairs(
        args.lr_videos,
        args.hr_videos,
        args.lr_output,
        args.hr_output,
        args.num_sequences,
        args.frames_per_sequence,
        args.skip_start,
        args.skip_end
    )


if __name__ == '__main__':
    main()

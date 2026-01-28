#!/usr/bin/env python3
"""
Noise Reduction - Remove background noise using spectral subtraction
"""

# Import required libraries
import numpy as np
from pydub import AudioSegment
import noisereduce as nr


def reduce_noise_spectral(audio_path, output_path, prop_decrease=0.8, stationary=True):
    """
    Apply spectral noise reduction to remove background noise
    
    This method uses frequency-based noise reduction which is much better
    at preserving soft speech compared to energy-based gating.
    
    Args:
        audio_path: Input audio file path
        output_path: Output audio file path
        prop_decrease: Proportion of noise to reduce (0.0-1.0, default 0.8 = 80%)
        stationary: True for constant background noise, False for changing noise
    
    Returns:
        None
    """
    
    print(f"\nApplying spectral noise reduction to: {audio_path}")
    
    # Load audio using pydub
    audio = AudioSegment.from_wav(audio_path)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # If stereo, convert to mono (take first channel)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples[:, 0]
    
    # Normalize to [-1, 1] range
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    samples = samples / max_val
    
    print(f"  Analyzing noise profile...")
    
    # Apply spectral noise reduction
    # The library automatically detects noise from quiet parts
    reduced_samples = nr.reduce_noise(
        y=samples,
        sr=audio.frame_rate,
        stationary=stationary,
        prop_decrease=prop_decrease
    )
    
    # Convert back to int16 and ensure proper shape
    reduced_samples = (reduced_samples * max_val).astype(np.int16)
    
    # Ensure the array is C-contiguous and properly aligned
    reduced_samples = np.ascontiguousarray(reduced_samples)
    
    # Create new AudioSegment from raw data
    cleaned_audio = AudioSegment(
        data=reduced_samples.tobytes(),
        sample_width=audio.sample_width,
        frame_rate=audio.frame_rate,
        channels=1
    )
    
    # Export cleaned audio
    cleaned_audio.export(output_path, format='wav')
    
    print(f"  ✓ Noise reduction applied (reduced by {prop_decrease*100:.0f}%)")
    print(f"  Saved: {output_path}\n")


# Backward compatibility wrapper (uses spectral reduction now)
def apply_noise_gate(audio_path, output_path, threshold_db=None, threshold_offset=8):
    """
    Legacy function for backward compatibility
    Now uses spectral noise reduction instead of energy gating
    
    Args:
        audio_path: Input audio file path
        output_path: Output audio file path
        threshold_db: Ignored (kept for compatibility)
        threshold_offset: Ignored (kept for compatibility)
    """
    # Use default spectral reduction settings
    reduce_noise_spectral(audio_path, output_path, prop_decrease=0.8, stationary=True)
    return -35.0, 15.0  # Dummy values for compatibility


# CLI interface
if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Remove background noise using spectral subtraction (frequency-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and reduce noise (80% reduction)
  python noise_reduction.py -f audio.wav
  
  # Gentle noise reduction (50%)
  python noise_reduction.py -f audio.wav --strength 0.5
  
  # Aggressive noise reduction (95%)
  python noise_reduction.py -f audio.wav --strength 0.95
  
  # Process entire folder
  python noise_reduction.py -i outputs/converted -o outputs/cleaned
        """
    )
    
    parser.add_argument('-f', '--file', type=str,
                        help='Process single audio file')
    parser.add_argument('-i', '--input', type=str, default='outputs/converted',
                        help='Input folder with WAV files')
    parser.add_argument('-o', '--output', type=str, default='outputs/cleaned',
                        help='Output folder for cleaned files')
    parser.add_argument('--strength', type=float, default=0.8,
                        help='Noise reduction strength 0.0-1.0 (default: 0.8 = 80%%)')
    parser.add_argument('--non-stationary', action='store_true',
                        help='Use for changing/non-constant background noise')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate strength
    strength = max(0.0, min(1.0, args.strength))
    stationary = not args.non_stationary
    
    # Process single file or folder
    if args.file:
        # Single file
        input_path = Path(args.file)
        output_path = output_dir / input_path.name
        reduce_noise_spectral(str(input_path), str(output_path), strength, stationary)
    else:
        # Batch processing
        input_dir = Path(args.input)
        audio_files = list(input_dir.glob('*.wav'))
        
        print(f"Found {len(audio_files)} WAV file(s) in {args.input}\n")
        print(f"{'='*60}")
        
        for audio_file in audio_files:
            output_path = output_dir / audio_file.name
            reduce_noise_spectral(str(audio_file), str(output_path), strength, stationary)
        
        print(f"{'='*60}")
        print(f"Noise reduction complete! Cleaned files saved to: {args.output}")
        print(f"{'='*60}\n")

#!/usr/bin/env python3
"""
Noise Reduction - Remove background noise using noise profiling
Analyzes first segment before speech to capture noise characteristics
"""

# Import required libraries
import numpy as np
from pydub import AudioSegment
import math


# Function to calculate RMS energy in dBFS
def calculate_rms_db(audio_segment):
    """Calculate RMS (Root Mean Square) energy in dBFS"""
    
    # Get audio samples as array
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Handle empty or silent segments
    if len(samples) == 0 or np.all(samples == 0):
        return -96.0  # Very quiet (near digital silence)
    
    # Calculate RMS
    rms = np.sqrt(np.mean(samples.astype(float) ** 2))
    
    # Convert to dBFS (decibels relative to full scale)
    # For 16-bit audio, max value is 32768
    max_value = float(2 ** (audio_segment.sample_width * 8 - 1))
    
    if rms > 0:
        dbfs = 20 * math.log10(rms / max_value)
    else:
        dbfs = -96.0  # Digital silence
    
    return dbfs


# Function to detect first speech spike
def detect_first_speech_spike(audio, chunk_duration_ms=100, spike_threshold_increase=10):
    """
    Find the first major spike in audio (where clear speech starts)
    
    Args:
        audio: AudioSegment object
        chunk_duration_ms: Size of chunks to analyze
        spike_threshold_increase: dB increase from baseline to consider a spike
    
    Returns:
        spike_position_ms: Position where first spike occurs (in milliseconds)
    """
    
    # Analyze first part of audio to find baseline
    baseline_chunks = []
    for i in range(0, min(3000, len(audio)), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        if len(chunk) > 0:
            baseline_chunks.append(calculate_rms_db(chunk))
    
    if not baseline_chunks:
        return 0
    
    # Find median of first 3 seconds as baseline
    baseline_energy = np.median(baseline_chunks)
    
    # Now scan for first major spike
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        if len(chunk) > 0:
            chunk_energy = calculate_rms_db(chunk)
            
            # Check if this is a significant spike above baseline
            if chunk_energy > baseline_energy + spike_threshold_increase:
                print(f"  First speech spike detected at: {i/1000:.2f}s (energy: {chunk_energy:.1f} dBFS)")
                return i
    
    # If no spike found, assume speech starts at 2 seconds
    print(f"  No clear spike detected, using default 2s")
    return 2000


# Function to profile noise from initial segment
def profile_noise_from_initial_segment(audio_path, profile_duration_ms=None, buffer_db=3):
    """
    Analyze audio before first speech spike to create noise profile
    
    Args:
        audio_path: Path to audio file
        profile_duration_ms: How much audio to use (None = auto-detect from spike)
        buffer_db: dB to add above noise ceiling (default 3)
    
    Returns:
        threshold_db: Recommended threshold based on noise profile
        noise_ceiling_db: Maximum noise level detected
        profile_duration: Duration of noise profile used
    """
    
    print(f"  Creating noise profile from initial segment...")
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Auto-detect profile duration if not specified
    if profile_duration_ms is None:
        spike_position = detect_first_speech_spike(audio)
        # Use everything before spike, or minimum 1 second
        profile_duration_ms = max(1000, spike_position - 200)  # 200ms before spike
    
    # Extract noise profile segment
    noise_segment = audio[:profile_duration_ms]
    
    # Analyze noise segment in small chunks
    chunk_duration_ms = 50
    noise_energies = []
    
    for i in range(0, len(noise_segment), chunk_duration_ms):
        chunk = noise_segment[i:i + chunk_duration_ms]
        if len(chunk) > 0:
            energy = calculate_rms_db(chunk)
            noise_energies.append(energy)
    
    if not noise_energies:
        print(f"  Warning: Could not analyze noise, using default threshold")
        return -40.0, -50.0, profile_duration_ms
    
    # Find noise characteristics
    noise_floor = np.percentile(noise_energies, 10)  # Quietest parts
    noise_ceiling = np.percentile(noise_energies, 90)  # Loudest noise parts
    noise_max = np.max(noise_energies)  # Absolute maximum
    
    # Set threshold slightly above the noisiest parts
    threshold_db = noise_max + buffer_db
    
    print(f"  Noise profile duration: {profile_duration_ms/1000:.2f}s")
    print(f"  Noise floor: {noise_floor:.1f} dBFS")
    print(f"  Noise ceiling: {noise_ceiling:.1f} dBFS")
    print(f"  Noise max: {noise_max:.1f} dBFS")
    print(f"  Threshold set to: {threshold_db:.1f} dBFS (noise max + {buffer_db} dB)")
    
    return threshold_db, noise_ceiling, profile_duration_ms


# Function to apply noise gate
def apply_noise_gate(audio_path, output_path, threshold_db=None, buffer_db=3, manual_profile_duration=None):
    """
    Apply noise gate using noise profiling from initial segment
    
    Args:
        audio_path: Input audio file path
        output_path: Output audio file path
        threshold_db: Manual threshold (dBFS). If None, auto-detect from noise profile
        buffer_db: dB buffer above noise ceiling (default 3)
        manual_profile_duration: Manual profile duration in seconds (None = auto-detect)
    
    Returns:
        threshold_used: The threshold that was applied
        removed_percentage: Percentage of audio muted
    """
    
    print(f"\nApplying noise-profiled gate to: {audio_path}")
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Create noise profile if threshold not manually specified
    if threshold_db is None:
        profile_duration_ms = manual_profile_duration * 1000 if manual_profile_duration else None
        threshold_db, noise_ceiling, profile_duration = profile_noise_from_initial_segment(
            audio_path, 
            profile_duration_ms, 
            buffer_db
        )
    else:
        print(f"  Using manual threshold: {threshold_db:.1f} dBFS")
    
    # Process audio in chunks
    chunk_duration_ms = 50  # 50ms chunks for smooth gating
    processed_chunks = []
    total_chunks = 0
    muted_chunks = 0
    
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        
        if len(chunk) == 0:
            continue
        
        total_chunks += 1
        
        # Calculate chunk energy
        chunk_energy = calculate_rms_db(chunk)
        
        # Apply gate
        if chunk_energy < threshold_db:
            # Below threshold - replace with silence
            silent_chunk = AudioSegment.silent(duration=len(chunk), frame_rate=audio.frame_rate)
            processed_chunks.append(silent_chunk)
            muted_chunks += 1
        else:
            # Above threshold - keep original
            processed_chunks.append(chunk)
    
    # Combine processed chunks
    if len(processed_chunks) > 0:
        cleaned_audio = processed_chunks[0]
        for chunk in processed_chunks[1:]:
            cleaned_audio += chunk
    else:
        cleaned_audio = audio
    
    # Export cleaned audio
    cleaned_audio.export(output_path, format='wav')
    
    # Calculate statistics
    removed_percentage = (muted_chunks / total_chunks * 100) if total_chunks > 0 else 0
    
    print(f"  ✓ Noise gate applied")
    print(f"  Muted: {removed_percentage:.1f}% of audio")
    print(f"  Saved: {output_path}\n")
    
    return threshold_db, removed_percentage


# CLI interface
if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Remove background noise using intelligent noise profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect noise from beginning (RECOMMENDED)
  python noise_reduction.py -f audio.wav
  
  # Manual threshold
  python noise_reduction.py -f audio.wav -t -35
  
  # Custom buffer (more/less aggressive)
  python noise_reduction.py -f audio.wav --buffer 5
  
  # Specify noise profile duration manually
  python noise_reduction.py -f audio.wav --profile 3
  
  # Process folder
  python noise_reduction.py -i outputs/converted -o outputs/cleaned
        """
    )
    
    parser.add_argument('-f', '--file', type=str,
                        help='Process single audio file')
    parser.add_argument('-i', '--input', type=str, default='outputs/converted',
                        help='Input folder with WAV files')
    parser.add_argument('-o', '--output', type=str, default='outputs/cleaned',
                        help='Output folder for cleaned files')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Manual threshold in dBFS (e.g., -35)')
    parser.add_argument('--buffer', type=float, default=3,
                        help='dB buffer above noise max (default: 3 dB)')
    parser.add_argument('--profile', type=float,
                        help='Noise profile duration in seconds (default: auto-detect from first spike)')

    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single file or folder
    if args.file:
        # Single file
        input_path = Path(args.file)
        output_path = output_dir / input_path.name
        apply_noise_gate(str(input_path), str(output_path), args.threshold, args.buffer, args.profile)
    else:
        # Batch processing
        input_dir = Path(args.input)
        audio_files = list(input_dir.glob('*.wav'))
        
        print(f"Found {len(audio_files)} WAV file(s) in {args.input}\n")
        print(f"{'='*60}")
        
        for audio_file in audio_files:
            output_path = output_dir / audio_file.name
            apply_noise_gate(str(audio_file), str(output_path), args.threshold, args.buffer, args.profile)
        
        print(f"{'='*60}")
        print(f"Noise reduction complete! Cleaned files saved to: {args.output}")
        print(f"{'='*60}\n")

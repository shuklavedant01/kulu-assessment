#!/usr/bin/env python3
"""
Noise Reduction - Remove background noise using VAD + energy-based filtering
Uses Voice Activity Detection to protect speech from being cut off
"""

# Import required libraries
import numpy as np
from pydub import AudioSegment
import math
import webrtcvad
import struct


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


# Function to detect speech using WebRTC VAD
def detect_speech_segments(audio, vad_aggressiveness=1):
    """
    Use Voice Activity Detection to find speech segments
    
    Args:
        audio: AudioSegment object
        vad_aggressiveness: 0-3, higher = more aggressive filtering (1 = balanced)
    
    Returns:
        List of tuples (start_ms, end_ms) indicating speech segments
    """
    
    print(f"  Running Voice Activity Detection (aggressiveness={vad_aggressiveness})...")
    
    # WebRTC VAD works with specific sample rates
    # We'll use 16kHz which is standard for speech
    if audio.frame_rate not in [8000, 16000, 32000, 48000]:
        audio = audio.set_frame_rate(16000)
    
    # VAD requires mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Initialize VAD
    vad = webrtcvad.Vad(vad_aggressiveness)
    
    # Process in 30ms frames (VAD requirement)
    frame_duration_ms = 30
    frame_duration = int(audio.frame_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample (16-bit)
    
    speech_segments = []
    is_speech_active = False
    speech_start = 0
    
    # Get raw audio data
    raw_data = audio.raw_data
    
    # Process each frame
    for i in range(0, len(raw_data), frame_duration):
        frame = raw_data[i:i + frame_duration]
        
        # Skip if frame is too short
        if len(frame) < frame_duration:
            break
        
        # Check if frame contains speech
        try:
            is_speech = vad.is_speech(frame, audio.frame_rate)
            
            current_time_ms = int(i / (audio.frame_rate * 2) * 1000)
            
            if is_speech and not is_speech_active:
                # Speech started
                speech_start = current_time_ms
                is_speech_active = True
            elif not is_speech and is_speech_active:
                # Speech ended
                speech_segments.append((speech_start, current_time_ms))
                is_speech_active = False
                
        except Exception as e:
            # If VAD fails on this frame, skip it
            pass
    
    # Handle case where speech continues to end of audio
    if is_speech_active:
        speech_segments.append((speech_start, len(audio)))
    
    print(f"  Detected {len(speech_segments)} speech segments")
    
    return speech_segments


# Function to check if timestamp is in speech segment
def is_in_speech_segment(time_ms, speech_segments):
    """Check if given timestamp falls within any speech segment"""
    for start, end in speech_segments:
        if start <= time_ms <= end:
            return True
    return False


# Function to detect noise floor
def detect_noise_floor(audio_path, chunk_duration_ms=100, percentile=15):
    """
    Analyze audio to find the noise floor (baseline background noise level)
    
    Args:
        audio_path: Path to audio file
        chunk_duration_ms: Size of chunks to analyze (milliseconds)
        percentile: Percentile to use for noise floor (default 15 = bottom 15%)
    
    Returns:
        noise_floor_db: Noise floor level in dBFS
    """
    
    print(f"  Analyzing noise floor...")
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    # Calculate energy for each chunk
    chunk_energies = []
    for chunk in chunks:
        energy_db = calculate_rms_db(chunk)
        chunk_energies.append(energy_db)
    
    # Find noise floor (15th percentile = typical background noise)
    noise_floor_db = np.percentile(chunk_energies, percentile)
    
    print(f"  Noise floor detected: {noise_floor_db:.1f} dBFS")
    
    return noise_floor_db


# Function to apply VAD-protected noise gate
def apply_noise_gate(audio_path, output_path, threshold_db=None, threshold_offset=8, vad_aggressiveness=1):
    """
    Apply noise gate with VAD protection - speech is NEVER cut
    
    Args:
        audio_path: Input audio file path
        output_path: Output audio file path
        threshold_db: Manual threshold (dBFS). If None, auto-detect
        threshold_offset: dB above noise floor for auto threshold (default 8)
        vad_aggressiveness: VAD sensitivity 0-3 (1 = balanced, 3 = very aggressive)
    
    Returns:
        threshold_used: The threshold that was applied
        removed_percentage: Percentage of audio muted
    """
    
    print(f"\nApplying VAD-protected noise gate to: {audio_path}")
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Detect speech segments using VAD
    speech_segments = detect_speech_segments(audio, vad_aggressiveness)
    
    # Auto-detect threshold if not provided
    if threshold_db is None:
        noise_floor = detect_noise_floor(audio_path)
        threshold_db = noise_floor + threshold_offset
        print(f"  Auto threshold: {threshold_db:.1f} dBFS (noise floor + {threshold_offset} dB)")
    else:
        print(f"  Manual threshold: {threshold_db:.1f} dBFS")
    
    # Process audio in chunks
    chunk_duration_ms = 50  # 50ms chunks for smooth gating
    processed_chunks = []
    total_chunks = 0
    muted_chunks = 0
    protected_chunks = 0
    
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        
        if len(chunk) == 0:
            continue
        
        total_chunks += 1
        current_time_ms = i
        
        # Check if this chunk is in a speech segment
        is_speech = is_in_speech_segment(current_time_ms, speech_segments)
        
        if is_speech:
            # ALWAYS keep speech chunks - VAD protection
            processed_chunks.append(chunk)
            protected_chunks += 1
        else:
            # Non-speech: apply energy gate
            chunk_energy = calculate_rms_db(chunk)
            
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
    protected_percentage = (protected_chunks / total_chunks * 100) if total_chunks > 0 else 0
    
    print(f"  ✓ Noise gate applied")
    print(f"  Speech protected: {protected_percentage:.1f}% (VAD)")
    print(f"  Noise muted: {removed_percentage:.1f}%")
    print(f"  Saved: {output_path}\n")
    
    return threshold_db, removed_percentage


# CLI interface
if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Remove background noise using VAD + energy-based filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect threshold with VAD protection
  python noise_reduction.py -f audio.wav
  
  # Manual threshold with gentle VAD
  python noise_reduction.py -f audio.wav -t -35 --vad 0
  
  # Aggressive VAD (more speech protection)
  python noise_reduction.py -f audio.wav --vad 3
  
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
    parser.add_argument('--offset', type=float, default=8,
                        help='Threshold offset above noise floor (default: 8 dB)')
    parser.add_argument('--vad', type=int, default=1, choices=[0, 1, 2, 3],
                        help='VAD aggressiveness: 0=gentle, 1=balanced (default), 2=aggressive, 3=very aggressive')

    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single file or folder
    if args.file:
        # Single file
        input_path = Path(args.file)
        output_path = output_dir / input_path.name
        apply_noise_gate(str(input_path), str(output_path), args.threshold, args.offset, args.vad)
    else:
        # Batch processing
        input_dir = Path(args.input)
        audio_files = list(input_dir.glob('*.wav'))
        
        print(f"Found {len(audio_files)} WAV file(s) in {args.input}\n")
        print(f"{'='*60}")
        
        for audio_file in audio_files:
            output_path = output_dir / audio_file.name
            apply_noise_gate(str(audio_file), str(output_path), args.threshold, args.offset, args.vad)
        
        print(f"{'='*60}")
        print(f"Noise reduction complete! Cleaned files saved to: {args.output}")
        print(f"{'='*60}\n")

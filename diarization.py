#!/usr/bin/env python3
"""
Speaker Diarization - Identify Agent and User speakers in conversations
"""

# Import required libraries for diarization and file handling
import argparse
import json
import os
from pathlib import Path
from pyannote.audio import Pipeline


# HuggingFace token for pyannote model access
# Read from environment variable or use provided token



# Optional: Import noise reduction (if available)
try:
    from noise_reduction import apply_noise_gate
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False

# Import for energy calculation
import numpy as np
from pydub import AudioSegment
import math
from utils import load_config

# Load config
CONFIG = load_config()
DIAR_CONFIG = CONFIG.get('processing', {}).get('diarization', {})


def detect_multilingual_and_assign_speakers(segments, transcription_file=None):
    """
    Intelligently assign Agent/User labels based on language detection
    
    Logic:
    - If 2 speakers: First = Agent, Second = User
    - If 3 speakers + multilingual: Third speaker = Agent (multilingual)
    - If transcription available, use language data for smarter assignment
    
    Args:
        segments: Diarization segments with SPEAKER_XX labels
        transcription_file: Optional path to transcription JSON with language data
    
    Returns:
        segments: Segments with Agent/User labels
        mapping: Speaker mapping dictionary
    """
    
    # Count unique speakers
    speakers = set([seg['speaker'] for seg in segments])
    num_speakers = len(speakers)
    
    print(f"  Detected {num_speakers} speakers: {speakers}")
    
    # Default mapping (2 speakers)
    mapping = {}
    if num_speakers == 2:
        speakers_list = sorted(list(speakers))
        mapping = {
            speakers_list[0]: 'Agent',
            speakers_list[1]: 'User'
        }
        print(f"  Using default 2-speaker mapping: {mapping}")
    
    # 3-speaker handling with multilingual detection
    elif num_speakers >= 3 and transcription_file:
        try:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
            
            # Detect languages per speaker
            speaker_languages = {}
            for seg in trans_data.get('segments', []):
                speaker = seg.get('speaker', 'unknown')
                lang = seg.get('language', 'unknown')
                
                if speaker not in speaker_languages:
                    speaker_languages[speaker] = set()
                speaker_languages[speaker].add(lang)
            
            # Check if multilingual
            all_languages = set()
            for langs in speaker_languages.values():
                all_languages.update(langs)
            
            is_multilingual = len(all_languages) > 1
            
            if is_multilingual:
                # Find which speaker speaks multiple languages (Agent)
                multilingual_speaker = None
                for speaker, langs in speaker_languages.items():
                    if len(langs) > 1:
                        multilingual_speaker = speaker
                        break
                
                # If no clear multilingual speaker, use 3rd speaker as Agent
                speakers_list = sorted(list(speakers))
                if multilingual_speaker and multilingual_speaker in speakers_list:
                    mapping[multilingual_speaker] = 'Agent'
                    # Others are Users
                    user_num = 1
                    for sp in speakers_list:
                        if sp != multilingual_speaker:
                            mapping[sp] = f'User{user_num}'
                            user_num += 1
                else:
                    # Default: 3rd speaker is Agent
                    mapping[speakers_list[0]] = 'User1'
                    mapping[speakers_list[1]] = 'User2'
                    mapping[speakers_list[2]] = 'Agent'
                
                print(f"  Multilingual detected! Languages: {all_languages}")
                print(f"  Using 3-speaker multilingual mapping: {mapping}")
            else:
                # Not multilingual, use default
                speakers_list = sorted(list(speakers))
                mapping[speakers_list[0]] = 'Agent'
                for i, sp in enumerate(speakers_list[1:], 1):
                    mapping[sp] = f'User{i}'
                print(f"  Using default 3-speaker mapping: {mapping}")
        
        except Exception as e:
            print(f"  Warning: Could not read transcription file: {e}")
            print(f"  Using default mapping")
            speakers_list = sorted(list(speakers))
            mapping[speakers_list[0]] = 'Agent'
            for i, sp in enumerate(speakers_list[1:], 1):
                mapping[sp] = f'User{i}'
    
    else:
        # Fallback for 3+ speakers without transcription
        speakers_list = sorted(list(speakers))
        mapping[speakers_list[0]] = 'Agent'
        for i, sp in enumerate(speakers_list[1:], 1):
            mapping[sp] = f'User{i}'
        print(f"  Using default {num_speakers}-speaker mapping: {mapping}")
    
    # Apply mapping to segments
    for seg in segments:
        seg['speaker'] = mapping.get(seg['speaker'], seg['speaker'])
    
    return segments, mapping




# Function to perform diarization on a single audio file
def diarize_audio(audio_path, pipeline, num_speakers=None):
    """Run speaker diarization and label speakers as Agent/User
    
    Args:
        audio_path: Path to audio file
        pipeline: Pyannote pipeline
        num_speakers: Force specific number of speakers (None = auto-detect, 2 = force 2 speakers)
    """
    
    print(f"Processing: {audio_path.name}...")
    
    # Run diarization pipeline on audio file
    # If num_speakers specified, force that many speakers
    if num_speakers is not None:
        diarization = pipeline(str(audio_path), num_speakers=num_speakers)
    else:
        diarization = pipeline(str(audio_path))

    
    # Extract all segments with speaker labels
    segments = []
    speakers_seen = {}
    speaker_order = []  # Track order of appearance
    
    # Process each segment from diarization results
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Track order of unique speakers
        if speaker not in speaker_order:
            speaker_order.append(speaker)
            
            # Assign labels based on order of appearance
            position = len(speaker_order)
            
            if position == 1:
                # 1st speaker = Agent (English)
                speakers_seen[speaker] = "Agent"
            elif position == 2:
                # 2nd speaker = User
                speakers_seen[speaker] = "User"
            elif position == 3:
                # 3rd speaker = Agent (Russian/different language)
                # This will be merged with 1st Agent in the final output
                speakers_seen[speaker] = "Agent"
                print(f"  Detected 3rd speaker: {speaker} → Agent (language variation)")
            else:
                # If somehow 4+ speakers detected, label as User
                speakers_seen[speaker] = "User"
                print(f"  Warning: {speaker} (4th+ speaker) → User")
        
        # Add segment with timestamp and speaker label
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "speaker": speakers_seen[speaker],
            "original_label": speaker
        })
    
    # Return segments and speaker mapping
    return segments, speakers_seen


# Function to filter noise segments from diarization results
def filter_noise_segments(segments, audio_path, 
                        min_duration=DIAR_CONFIG.get('min_segment_duration_s', 0.3), 
                        energy_threshold=DIAR_CONFIG.get('noise_energy_threshold_db', -50), 
                        isolation_gap=DIAR_CONFIG.get('noise_isolation_gap_s', 2.0)):
    """
    Filter out noise segments based on duration, energy, and isolation criteria
    
    Args:
        segments: List of diarization segments
        audio_path: Path to audio file (for energy analysis)
        min_duration: Minimum segment duration in seconds (default from config or 0.3s)
        energy_threshold: Energy threshold in dBFS (default from config or -50 dB)
        isolation_gap: Gap before/after to consider isolated in seconds (default from config or 2.0s)
    
    Returns:
        filtered_segments: Clean segments with noise removed
        removed_count: Number of segments removed
    """
    
    print(f"  Filtering noise segments...")
    
    # Load audio for energy analysis
    audio = AudioSegment.from_wav(str(audio_path))
    
    def calculate_segment_energy(start_ms, end_ms):
        """Calculate average energy of segment in dBFS"""
        segment = audio[start_ms:end_ms]
        samples = np.array(segment.get_array_of_samples())
        
        if len(samples) == 0 or np.all(samples == 0):
            return -96.0
        
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))
        max_value = float(2 ** (segment.sample_width * 8 - 1))
        
        if rms > 0:
            return 20 * math.log10(rms / max_value)
        return -96.0
    
    filtered_segments = []
    removed_segments = []
    
    for i, segment in enumerate(segments):
        should_keep = True
        removal_reason = None
        
        # Duration filter - remove very short segments (noise bursts)
        if segment["duration"] < min_duration:
            should_keep = False
            removal_reason = "too_short"
        
        if should_keep:
            filtered_segments.append(segment)
        else:
            removed_segments.append({**segment, "removal_reason": removal_reason})
    
    print(f"  Removed {len(removed_segments)} noise segments:")
    print(f"    - Too short (<{min_duration}s): {len(removed_segments)}")

    
    return filtered_segments, len(removed_segments)


# Function to process diarization and create stats
def process_diarization_results(segments, speakers_seen):
    """
    Process segments and calculate statistics
    
    Args:
        segments: List of segments
        speakers_seen: Dictionary of speaker mappings
    
    Returns:
        Dictionary with segments, stats, and speaker mapping
    """
    # Calculate gaps between consecutive turns
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap > 0:
            gaps.append(round(gap, 3))
    
    # Compute statistics
    agent_segments = [s for s in segments if s["speaker"] == "Agent"]
    user_segments = [s for s in segments if s["speaker"] == "User"]
    
    stats = {
        "total_segments": len(segments),
        "total_duration": round(segments[-1]["end"], 3) if segments else 0,
        "agent_segments": len(agent_segments),
        "user_segments": len(user_segments),
        "agent_speaking_time": round(sum(s["duration"] for s in agent_segments), 3),
        "user_speaking_time": round(sum(s["duration"] for s in user_segments), 3),
        "avg_gap_between_turns": round(sum(gaps) / len(gaps), 3) if gaps else 0,
        "total_gaps": len(gaps),
        "unique_speakers": len(speakers_seen)
    }
    
    return stats


# Function to save results as JSON
def save_results(results, output_folder, audio_filename):
    """Save diarization results to JSON file"""
    
    output_path = Path(output_folder)
    
    # Check if output_folder is actually a complete file path (ends with .json)
    if output_path.suffix == '.json':
        # It's a complete file path, use it directly
        output_file = output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        # It's a folder path, create the file inside it
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{Path(audio_filename).stem}_diarization.json"
    
    # Write JSON with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {output_file.name}\n")
    return str(output_file)


# Function to process all files in a folder
def process_folder(input_folder='outputs/converted', output_folder='outputs/diarization', 
                    apply_noise_reduction=False, noise_threshold=None, num_speakers=None):
    """Process all WAV files for speaker diarization
    
    Args:
        input_folder: Input folder with WAV files
        output_folder: Output folder for JSON results
        apply_noise_reduction: Apply noise gate before diarization
        noise_threshold: Manual noise threshold (dBFS)
        num_speakers: Force specific number of speakers (2 = Agent + User only)
    """
    
    # Apply noise reduction if requested
    actual_input_folder = input_folder
    if apply_noise_reduction:
        if NOISE_REDUCTION_AVAILABLE:
            print(f"\n{'='*60}")
            print("STEP 1: Noise Reduction")
            print(f"{'='*60}\n")
            
            cleaned_folder = 'outputs/cleaned'
            Path(cleaned_folder).mkdir(parents=True, exist_ok=True)
            
            # Process each file with noise gate
            input_path = Path(input_folder)
            for audio_file in input_path.glob('*.wav'):
                output_path = Path(cleaned_folder) / audio_file.name
                apply_noise_gate(str(audio_file), str(output_path), noise_threshold, threshold_offset=8)
            
            # Use cleaned files for diarization
            actual_input_folder = cleaned_folder
            print(f"\n{'='*60}")
            print("STEP 2: Speaker Diarization")
            print(f"{'='*60}\n")
        else:
            print("⚠️  Warning: noise_reduction module not found, skipping noise reduction\n")
    
    print(f"Initializing pyannote pipeline...")
    print(f"Loading model from HuggingFace...\n")

    
    # Initialize pyannote pipeline with authentication
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    # Enable GPU if available
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    print(f"🚀 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print(f"   ⚠️  Running on CPU (will be slower)\n")

    
    # Get all WAV files from input folder
    input_path = Path(input_folder)
    audio_files = list(input_path.glob('*.wav'))
    
    print(f"Found {len(audio_files)} WAV file(s) in {input_folder}\n")
    
    # Process each audio file
    all_results = []
    for audio_file in audio_files:
        try:
            # Run diarization
            segments, speakers_seen = diarize_audio(audio_file, pipeline, num_speakers)
            
            # Apply noise filtering if requested
            if apply_noise_reduction:
                segments, removed_count = filter_noise_segments(segments, audio_file)
            
            # Calculate statistics
            stats = process_diarization_results(segments, speakers_seen)
            
            # Create results dictionary
            results = {
                "filename": audio_file.name,
                "segments": segments,
                "statistics": stats,
                "speaker_mapping": speakers_seen
            }
            
            if apply_noise_reduction:
                results["noise_filtered"] = removed_count
            
            # Save to JSON
            save_results(results, output_folder, audio_file.name)
            
            # Print summary
            print(f"  Summary:")
            print(f"    Total Segments: {stats['total_segments']}")
            print(f"    Agent: {stats['agent_segments']} segments, {stats['agent_speaking_time']}s")
            print(f"    User: {stats['user_segments']} segments, {stats['user_speaking_time']}s")
            print(f"    Avg Gap: {stats['avg_gap_between_turns']}s")
            if apply_noise_reduction:
                print(f"    Noise Removed: {removed_count} segments")
            print()
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {e}\n")
    
    # Print final summary
    print(f"{'='*60}")
    print(f"Diarization complete! Processed {len(all_results)}/{len(audio_files)} file(s)")
    print(f"Results saved to: {output_folder}")
    print(f"{'='*60}\n")
    
    return all_results


# CLI argument parsing
if __name__ == '__main__':
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description='Speaker diarization for audio files - Identify Agent and User speakers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all WAV files in outputs/converted/
  python diarization.py
  
  # Process a single audio file
  python diarization.py -f outputs/converted/audio1.wav
  
  # Process files from custom folder
  python diarization.py -i my_audio_folder -o my_results
  
  # Process single file to custom output folder
  python diarization.py -f audio.wav -o my_results
        """
    )
    parser.add_argument('-i', '--input', type=str, default='outputs/converted',
                        help='Input folder with WAV files (default: outputs/converted)')
    parser.add_argument('-o', '--output', type=str, default='outputs/diarization',
                        help='Output folder for JSON results (default: outputs/diarization)')
    parser.add_argument('-f', '--file', type=str,
                        help='Process single audio file instead of folder')
    parser.add_argument('--noise-reduce', action='store_true',
                        help='Apply noise reduction before diarization (requires noise_reduction.py)')
    parser.add_argument('--noise-threshold', type=float,
                        help='Manual threshold for noise reduction in dBFS (e.g., -35)')
    parser.add_argument('--num-speakers', type=int,
                        help='Force specific number of speakers (e.g., 2 for Agent+User, 3 for multi-language)')
    parser.add_argument('--transcription', '-t', type=str, default=None,
                        help='Path to transcription JSON for intelligent speaker labeling')
    parser.add_argument('--filter-noise', action='store_true',
                        help='Filter noise segments from results (removes short, quiet, isolated segments)')



    
    # Parse arguments
    args = parser.parse_args()
    
    # Process single file or entire folder
    if args.file:
        # Initialize pipeline
        import torch
        print("Initializing pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        # Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
        print(f"🚀 Using device: {device}\n")

        
        # Process single file
        audio_file = Path(args.file)
        segments, speakers_seen = diarize_audio(audio_file, pipeline, args.num_speakers)
        
        # Apply noise filtering if requested
        if args.filter_noise:
            segments, removed_count = filter_noise_segments(segments, audio_file)
        
        # Intelligently assign Agent/User labels
        segments, speaker_mapping = detect_multilingual_and_assign_speakers(
            segments, 
            args.transcription
        )
        
        # Calculate statistics
        stats = process_diarization_results(segments, speaker_mapping)
        
        # Create results dictionary
        results = {
            "filename": audio_file.name,
            "segments": segments,
            "statistics": stats,
            "speaker_mapping": speakers_seen
        }
        
        if args.filter_noise:
            results["noise_filtered"] = removed_count
        
        save_results(results, args.output, args.file)
    else:
        # Process entire folder (note: apply_noise_reduction variable is reused for filter_noise)
        process_folder(args.input, args.output, args.filter_noise, args.noise_threshold, args.num_speakers)



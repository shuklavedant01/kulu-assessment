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



# Function to perform diarization on a single audio file
def diarize_audio(audio_path, pipeline):
    """Run speaker diarization and label speakers as Agent/User"""
    
    print(f"Processing: {audio_path.name}...")
    
    # Run diarization pipeline on audio file
    diarization = pipeline(str(audio_path))
    
    # Extract all segments with speaker labels
    segments = []
    speakers_seen = {}
    first_speaker = None
    
    # Process each segment from diarization results
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Track first speaker to label as Agent
        if first_speaker is None:
            first_speaker = speaker
            speakers_seen[speaker] = "Agent"
        elif speaker not in speakers_seen:
            # All other speakers labeled as User
            speakers_seen[speaker] = "User"
        
        # Add segment with timestamp and speaker label
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "speaker": speakers_seen[speaker],
            "original_label": speaker
        })
    
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
    
    # Return complete diarization results
    return {
        "filename": audio_path.name,
        "segments": segments,
        "statistics": stats,
        "speaker_mapping": speakers_seen
    }


# Function to save results as JSON
def save_results(results, output_folder, audio_filename):
    """Save diarization results to JSON file"""
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = output_path / f"{Path(audio_filename).stem}_diarization.json"
    
    # Write JSON with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {output_file.name}\n")
    return str(output_file)


# Function to process all files in a folder
def process_folder(input_folder='outputs/converted', output_folder='outputs/diarization'):
    """Process all WAV files for speaker diarization"""
    
    print(f"\nInitializing pyannote pipeline...")
    print(f"Loading model from HuggingFace...\n")
    
    # Initialize pyannote pipeline with authentication
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    # Get all WAV files from input folder
    input_path = Path(input_folder)
    audio_files = list(input_path.glob('*.wav'))
    
    print(f"Found {len(audio_files)} WAV file(s) in {input_folder}\n")
    
    # Process each audio file
    all_results = []
    for audio_file in audio_files:
        try:
            # Run diarization
            results = diarize_audio(audio_file, pipeline)
            
            # Save to JSON
            save_results(results, output_folder, audio_file.name)
            
            # Print summary
            stats = results['statistics']
            print(f"  Summary:")
            print(f"    Total Segments: {stats['total_segments']}")
            print(f"    Agent: {stats['agent_segments']} segments, {stats['agent_speaking_time']}s")
            print(f"    User: {stats['user_segments']} segments, {stats['user_speaking_time']}s")
            print(f"    Avg Gap: {stats['avg_gap_between_turns']}s\n")
            
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

    
    # Parse arguments
    args = parser.parse_args()
    
    # Process single file or entire folder
    if args.file:
        # Initialize pipeline
        print("Initializing pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        # Process single file
        results = diarize_audio(Path(args.file), pipeline)
        save_results(results, args.output, args.file)
    else:
        # Process entire folder
        process_folder(args.input, args.output)

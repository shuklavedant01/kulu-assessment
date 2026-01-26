#!/usr/bin/env python3
"""
Audio Analyzer - Check audio properties and verify quality
"""

# Import required libraries for audio analysis
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np


# Function to analyze a single audio file
def analyze_audio(audio_path):
    """Analyze audio file and return properties"""
    
    # Load the audio file
    audio = AudioSegment.from_file(str(audio_path))
    
    # Get basic audio properties
    duration_sec = len(audio) / 1000.0  # Convert milliseconds to seconds
    sample_rate = audio.frame_rate
    channels = audio.channels
    bit_depth = audio.sample_width * 8  # Convert bytes to bits
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    
    # Check if audio has actual content (detect non-silent segments)
    # A silent threshold of -40 dBFS is used (lower = more silent)
    nonsilent_ranges = detect_nonsilent(
        audio, 
        min_silence_len=500,  # Minimum silence length in ms
        silence_thresh=-40    # Silence threshold in dBFS
    )
    has_content = len(nonsilent_ranges) > 0
    content_percentage = sum(end - start for start, end in nonsilent_ranges) / len(audio) * 100
    
    # Convert to numpy array to check for corruption
    samples = np.array(audio.get_array_of_samples())
    is_corrupted = len(samples) == 0 or np.all(samples == 0)
    
    # Return analysis results as dictionary
    return {
        'filename': audio_path.name,
        'duration_sec': duration_sec,
        'sample_rate': sample_rate,
        'channels': channels,
        'bit_depth': bit_depth,
        'file_size_mb': file_size_mb,
        'has_content': has_content,
        'content_percentage': content_percentage,
        'is_corrupted': is_corrupted,
        'nonsilent_segments': len(nonsilent_ranges)
    }


# Function to print analysis results
def print_analysis(results):
    """Print analysis results in a readable format"""
    
    print(f"\n{'='*60}")
    print(f"File: {results['filename']}")
    print(f"{'='*60}")
    
    # Basic properties
    print(f"\n📊 Audio Properties:")
    print(f"  Duration:     {results['duration_sec']:.2f} seconds ({results['duration_sec']/60:.2f} minutes)")
    print(f"  Sample Rate:  {results['sample_rate']} Hz")
    print(f"  Channels:     {results['channels']} ({'Mono' if results['channels'] == 1 else 'Stereo'})")
    print(f"  Bit Depth:    {results['bit_depth']} bits")
    print(f"  File Size:    {results['file_size_mb']:.2f} MB")
    
    # Quality check
    print(f"\n✓ Quality Check:")
    print(f"  Corrupted:    {'❌ Yes' if results['is_corrupted'] else '✅ No'}")
    print(f"  Has Content:  {'✅ Yes' if results['has_content'] else '❌ No'}")
    print(f"  Content:      {results['content_percentage']:.1f}% of file")
    print(f"  Speech/Sound: {results['nonsilent_segments']} segments detected")


# Function to analyze multiple files in a folder
def analyze_folder(folder_path):
    """Analyze all audio files in a folder"""
    
    # Get all audio files
    audio_path = Path(folder_path)
    audio_files = [f for f in audio_path.glob('*') if f.suffix.lower() in 
                   ['.mp3', '.wav', '.ogg', '.oga', '.flac', '.m4a', '.aac']]
    
    print(f"Found {len(audio_files)} audio file(s) in {folder_path}\n")
    
    # Analyze each file
    all_results = []
    for audio_file in audio_files:
        try:
            print(f"Analyzing: {audio_file.name}...")
            results = analyze_audio(audio_file)
            print_analysis(results)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error analyzing {audio_file.name}: {e}")
    
    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"Summary: {len(all_results)} file(s) analyzed")
        total_duration = sum(r['duration_sec'] for r in all_results)
        print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"{'='*60}\n")
    
    return all_results


# CLI argument parsing
if __name__ == '__main__':
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Analyze audio files')
    parser.add_argument('-f', '--file', type=str,
                        help='Single audio file to analyze')
    parser.add_argument('-d', '--dir', type=str, default='audio_files',
                        help='Directory containing audio files (default: audio_files)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Analyze single file or entire directory
    if args.file:
        print(f"\nAnalyzing single file: {args.file}")
        results = analyze_audio(Path(args.file))
        print_analysis(results)
    else:
        analyze_folder(args.dir)

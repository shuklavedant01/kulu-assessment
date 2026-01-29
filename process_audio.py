#!/usr/bin/env python3
"""
Complete Audio Processing Pipeline
Organizes all outputs in outputs/{audio_name}/
"""

import argparse
import subprocess
from pathlib import Path
import json


def detect_languages_in_transcription(transcription_file):
    """
    Detect unique languages in transcription file
    
    Args:
        transcription_file: Path to transcription JSON
    
    Returns:
        set: Set of unique language codes detected
    """
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        languages = set()
        for seg in data.get('segments', []):
            lang = seg.get('language')
            if lang:
                languages.add(lang)
        
        return languages
    except Exception as e:
        print(f"Warning: Could not detect languages from transcription: {e}")
        return set()




def process_audio(audio_file, whisper_model='large'):
    """
    Run complete pipeline for a single audio file
    All outputs go to outputs/{audio_name}/
    
    Args:
        audio_file: Path to audio file
        whisper_model: Whisper model size (tiny/base/small/medium/large)
    """
    
    audio_path = Path(audio_file)
    audio_name = audio_path.stem  # e.g., "audio1"
    
    # Create output directory structure
    output_base = Path('outputs') / audio_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {audio_name}")
    print(f"Output directory: {output_base}")
    print(f"{'='*70}\n")
    
    # Paths for intermediate files
    wav_file = output_base / f"{audio_name}.wav"
    diarization_file = output_base / f"{audio_name}_diarization.json"
    transcription_file = output_base / f"{audio_name}_complete.json"
    cutouts_dir = output_base / "cutouts"
    latency_dir = output_base / "latency"
    
    cutouts_dir.mkdir(exist_ok=True)
    latency_dir.mkdir(exist_ok=True)
    
    # Step 1: Convert to WAV (using pydub directly)
    print("STEP 1: Converting to WAV (16kHz mono)")
    print("-"*70)
    
    try:
        from pydub import AudioSegment
        import shutil
        
        # Clean up if wav_file exists as directory (from old audio_converter bug)
        if wav_file.exists() and wav_file.is_dir():
            print(f"  Removing corrupted directory: {wav_file}")
            shutil.rmtree(wav_file)
        
        # Load audio
        print(f"Loading: {audio_path.name}...")
        audio = AudioSegment.from_file(str(audio_path))
        
        # Convert to mono if needed
        if audio.channels > 1:
            print(f"  Converting {audio.channels} channels → 1 (mono)")
            audio = audio.set_channels(1)
        
        # Resample to 16kHz
        if audio.frame_rate != 16000:
            print(f"  Resampling {audio.frame_rate}Hz → 16000Hz")
            audio = audio.set_frame_rate(16000)
        
        # Export as WAV
        print(f"  Saving to: {wav_file}")
        audio.export(str(wav_file), format='wav')
        
        print(f"✓ Conversion complete!")
        
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        raise
    
    print(f"✓ Saved: {wav_file}\n")
    
    # Step 2: Transcription (FIRST - to get language info)
    print("STEP 2: Transcription (Whisper)")
    print("-"*70)
    cmd = [
        'python', 'timestamp_transcription.py',
        '--file', str(wav_file),
        '--model', whisper_model,
        '--output', str(transcription_file)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Saved: {transcription_file}\n")
    
    # Detect languages to determine speaker count
    print("Detecting languages for automatic speaker count...")
    languages = detect_languages_in_transcription(transcription_file)
    
    if len(languages) > 1:
        num_speakers = 3
        print(f"  Multilingual detected: {languages} → Setting num_speakers=3")
    else:
        num_speakers = 2
        print(f"  Monolingual detected: {languages} → Setting num_speakers=2")
    print()
    
    # Step 3: Speaker Diarization (SECOND - uses language info)
    print("STEP 3: Speaker Diarization (with intelligent Agent mapping)")
    print("-"*70)
    
    # Clean up if diarization_file exists as directory
    if diarization_file.exists() and diarization_file.is_dir():
        print(f"  Removing corrupted directory: {diarization_file}")
        import shutil
        shutil.rmtree(diarization_file)
    
    cmd = [
        'python', 'diarization.py',
        '--file', str(wav_file),
        '--transcription', str(transcription_file),
        '--num-speakers', str(num_speakers),  # Automatic based on language
        '--output', str(diarization_file)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Saved: {diarization_file}\n")
    
    # Step 4: Cutout Detection
    print("STEP 4: Cutout Detection")
    print("-"*70)
    cutout_output = cutouts_dir / f"{audio_name}_cutouts.json"
    cmd = [
        'python', 'cutout_detector_adaptive.py',
        '-t', str(transcription_file),
        '-d', str(diarization_file),
        '-o', str(cutouts_dir)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Saved: {cutout_output}\n")
    
    # Step 5: Latency Analysis
    print("STEP 5: Latency Analysis")
    print("-"*70)
    latency_output = latency_dir / f"{audio_name}_latency.json"
    cmd = [
        'python', 'latency_analyzer.py',
        '--file', str(transcription_file),
        '--output', str(latency_dir)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Saved: {latency_output}\n")
    
    # Step 6: Visualization
    print("STEP 6: Waveform Visualization")
    print("-"*70)
    viz_dir = output_base / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    cmd = [
        'python', 'visualizer.py',
        '--file', str(wav_file),
        '--output', str(viz_dir)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Saved visualizations to: {viz_dir}\n")
    
    # Create summary
    summary = {
        'audio_name': audio_name,
        'source_file': str(audio_path.absolute()),
        'output_directory': str(output_base.absolute()),
        'files': {
            'wav': str(wav_file),
            'diarization': str(diarization_file),
            'transcription': str(transcription_file),
            'cutouts': str(cutout_output),
            'latency': str(latency_output)
        }
    }
    
    summary_file = output_base / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"{'='*70}")
    print(f"✅ COMPLETE! All outputs in: {output_base}")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process audio file through complete pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio1
  python process_audio.py --audio "audio_files/audio1.mp3"
  
  # Process with language settings
  python process_audio.py --audio "audio_files/audio2.mp3" --source-lang "pl" --target-lang "en"
  
  # All outputs go to: outputs/audio1/, outputs/audio2/, etc.
        """
    )
    
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--model', type=str, default='large',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: large)')
    
    args = parser.parse_args()
    
    process_audio(args.audio, args.model)

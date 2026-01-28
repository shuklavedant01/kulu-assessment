#!/usr/bin/env python3
"""
Complete Pipeline - Audio to Final Merged Results
Runs the complete workflow: Diarization → Transcription (both versions) → Merge
"""

import subprocess
import json
from pathlib import Path
import argparse


def run_complete_pipeline(audio_file, whisper_model='large', output_base='outputs', force_speakers=None):
    """
    Run complete audio processing pipeline with SMART AUTO-DETECTION
    
    Flow:
    1. Run Whisper transcription (Pass 1 + Pass 2)
    2. Detect if multilingual from Pass 2 results
    3. Auto-decide speaker count: multilingual → 3, single → 2
    4. Run diarization with smart speaker count
    5. Merge both versions with diarization
    
    Args:
        audio_file: Path to audio file (WAV)
        whisper_model: Whisper model size
        output_base: Base output directory
        force_speakers: Override auto-detection (None for auto)
    
    Returns:
        Paths to final output files
    """
    
    audio_path = Path(audio_file)
    audio_name = audio_path.stem
    
    print(f"\n{'='*70}")
    print(f"SMART AUTO-DETECTION PIPELINE")
    print(f"{'='*70}")
    print(f"Audio: {audio_file}")
    print(f"Whisper Model: {whisper_model}")
    print(f"{'='*70}\n")
    
    # Step 1: Whisper Transcription (generates multilingual detection)
    print(f"📍 STEP 1: Whisper Transcription with Language Detection")
    print(f"{'-'*70}")
    
    trans_cmd = [
        'python', 'timestamp_transcription.py',
        '-f', str(audio_file),
        '--model', whisper_model
    ]
    
    result = subprocess.run(trans_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Transcription failed!")
        return None
    
    original_json = Path(f'{output_base}/transcription/{audio_name}_timestamp_transcription.json')
    translated_json = Path(f'{output_base}/transcription/{audio_name}_timestamp_transcription_translated.json')
    
    # Step 2: Read transcription to detect multilingual
    print(f"\n📍 STEP 2: Smart Speaker Count Detection")
    print(f"{'-'*70}")
    
    with open(original_json, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    
    is_multilingual = transcription_data.get('is_multilingual', False)
    detected_languages = transcription_data.get('languages_detected', [])
    
    # Auto-decide speaker count
    if force_speakers is not None:
        num_speakers = force_speakers
        print(f"🔧 User override: {num_speakers} speakers (forced)")
    elif is_multilingual:
        num_speakers = 3
        print(f"✓ Multilingual detected: {', '.join(detected_languages)}")
        print(f"✓ Auto-setting: 3 speakers (Agent language-switching mode)")
    else:
        num_speakers = 2
        print(f"✓ Single language detected: {', '.join(detected_languages)}")
        print(f"✓ Auto-setting: 2 speakers (standard mode)")
    
    print(f"\n📍 STEP 3: Speaker Diarization ({num_speakers} speakers)")
    print(f"{'-'*70}")

    
    
    # Run diarization with smart speaker count
    diar_cmd = ['python', 'diarization.py', '-f', str(audio_file), '--filter-noise', '--num-speakers', str(num_speakers)]
    
    result = subprocess.run(diar_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Diarization failed!")
        return None
    
    diarization_json = Path(f'{output_base}/diarization/{audio_name}_diarization.json')
    print(f"✓ Diarization complete: {diarization_json}\n")
    
    # Step 4: Merge both transcription versions with diarization
    print(f"📍 STEP 4: Merge with Speaker Labels (Dual Output)")
    print(f"{'-'*70}")

        'python', 'merge_results.py',
        '-t', str(original_json),
        '-d', str(diarization_json),
        '-o', f'{output_base}/final/{audio_name}_complete.json'
    ]
    
    result = subprocess.run(merge_cmd_original, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Merge (original) failed!")
        return None
    
    # Merge translated
    merge_cmd_translated = [
        'python', 'merge_results.py',
        '-t', str(translated_json),
        '-d', str(diarization_json),
        '-o', f'{output_base}/final/{audio_name}_complete_translated.json'
    ]
    
    result = subprocess.run(merge_cmd_translated, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Merge (translated) failed!")
        return None
    
    final_original = Path(f'{output_base}/final/{audio_name}_complete.json')
    final_translated = Path(f'{output_base}/final/{audio_name}_complete_translated.json')
    
    print(f"✓ Merge complete\n")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Final Outputs:")
    print(f"  📄 Original (multilingual): {final_original}")
    print(f"  🌐 Translated (all English): {final_translated}")
    print(f"\n{'='*70}\n")
    
    return {
        'original': str(final_original),
        'translated': str(final_translated)
    }


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete audio processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect speakers)
  python pipeline.py -f audio.wav
  
  # Force 3-speaker mode (for multilingual Agent)
  python pipeline.py -f audio.wav --num-speakers 3
  
  # Use better Whisper model
  python pipeline.py -f audio.wav --model large --num-speakers 3
  
  # Process converted audio
  python pipeline.py -f outputs/converted/audio1.wav --model large
        """
    )
    
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Audio file (WAV) to process')
    parser.add_argument('--force-speakers', type=int,
                        help='Override auto-detection (e.g., 3 for multilingual). Leave empty for smart auto-detect.')
    parser.add_argument('--model', type=str, default='large',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: large, recommended for multilingual)')
    parser.add_argument('--output-base', type=str, default='outputs',
                        help='Base output directory (default: outputs)')
    
    args = parser.parse_args()
    
    # Run pipeline with smart auto-detection
    results = run_complete_pipeline(
        args.file,
        args.model,
        args.output_base,
        args.force_speakers
    )
    
    if results:
        print("🎉 Success! Check the final outputs above.")
    else:
        print("❌ Pipeline failed. Check errors above.")

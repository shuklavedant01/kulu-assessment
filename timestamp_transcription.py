#!/usr/bin/env python3
"""
Timestamp-First Transcription - Use Whisper timestamps to re-transcribe segments
This preserves multilingual content while maintaining accurate timestamps
"""

import whisper
import json
from pathlib import Path
from pydub import AudioSegment
import argparse


def get_timestamps_from_whisper(audio_path, model_name='base'):
    """
    First pass: Get accurate timestamps from Whisper (may translate)
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
    
    Returns:
        Segments with accurate timestamps
    """
    
    print(f"\n{'='*60}")
    print(f"PASS 1: Getting accurate timestamps from Whisper")
    print(f"{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Model: {model_name}\n")
    
    # Load model
    model = whisper.load_model(model_name)
    
    # Transcribe to get timestamps AND translate to English
    print(f"Processing for timestamp detection...")
    result = model.transcribe(
        str(audio_path),
        task='translate',  # Explicitly translate all languages to English
        word_timestamps=True,
        verbose=True  # Show progress to ensure completion
    )
    
    print(f"✓ Found {len(result['segments'])} segments with timestamps")
    
    # Filter out hallucinated segments (noise misdetected as speech)
    filtered_segments = []
    skipped_hallucinations = 0
    
    for seg in result['segments']:
        # Skip segments with high no-speech probability (likely hallucinations)
        no_speech_prob = seg.get('no_speech_probability', 0)
        if no_speech_prob > 0.6:  # Threshold: 60% confidence it's NOT speech
            skipped_hallucinations += 1
            continue
        filtered_segments.append(seg)
    
    print(f"  Total duration: {result['segments'][-1]['end']:.1f}s")
    print(f"  Filtered out {skipped_hallucinations} hallucinated segments (noise)\n")
    
    return filtered_segments


def retranscribe_segments_with_language(audio_path, segments, model_name='base', language_map=None):
    """
    Second pass: Re-transcribe each segment with correct language to prevent translation
    
    Args:
        audio_path: Path to audio file
        segments: Segments from first pass (with timestamps)
        model_name: Whisper model size
        language_map: Function or dict to determine language per segment
    
    Returns:
        Segments with correct language transcriptions
    """
    
    print(f"{'='*60}")
    print(f"PASS 2: Re-transcribing segments with language detection")
    print(f"{'='*60}\n")
    
    # Load audio
    audio = AudioSegment.from_wav(str(audio_path))
    
    # Load model
    model = whisper.load_model(model_name)
    
    retranscribed_segments = []
    languages_detected = set()
    
    for i, segment in enumerate(segments):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        
        # Extract segment audio
        segment_audio = audio[start_ms:end_ms]
        
        # Save temporary segment
        temp_file = f"temp_segment_{i}.wav"
        segment_audio.export(temp_file, format='wav')
        
        # Determine language for this segment
        if language_map:
            if callable(language_map):
                forced_lang = language_map(segment)
            else:
                forced_lang = language_map.get(i, None)
        else:
            forced_lang = None
        
        # Re-transcribe this segment with language hint
        seg_result = model.transcribe(
            temp_file,
            language=forced_lang,  # Force language to prevent translation
            task='transcribe',
            verbose=False
        )
        
        detected_lang = seg_result.get('language', 'unknown')
        languages_detected.add(detected_lang)
        lang_name = whisper.tokenizer.LANGUAGES.get(detected_lang, detected_lang)
        
        # Store result
        retranscribed_segments.append({
            'start': round(segment['start'], 3),
            'end': round(segment['end'], 3),
            'duration': round(segment['end'] - segment['start'], 3),
            'text': seg_result['text'].strip(),
            'language': detected_lang,
            'language_name': lang_name
        })
        
        print(f"  Segment {i+1}/{len(segments)}: [{segment['start']:.1f}s - {segment['end']:.1f}s] → {lang_name}")
        
        # Clean up
        Path(temp_file).unlink()
    
    print(f"\n✓ Re-transcribed {len(retranscribed_segments)} segments")
    print(f"  Languages detected: {', '.join(sorted(languages_detected))}\n")
    
    return retranscribed_segments, list(sorted(languages_detected))


def merge_with_diarization(transcribed_segments, diarization_json):
    """
    Merge transcription segments with diarization speaker labels
    
    Args:
        transcribed_segments: Segments with transcriptions
        diarization_json: Path to diarization JSON with speaker labels
    
    Returns:
        Merged segments with speaker + transcription + language
    """
    
    print(f"Merging with diarization...")
    
    # Load diarization
    with open(diarization_json, 'r', encoding='utf-8') as f:
        diarization = json.load(f)
    
    merged = []
    
    # Match transcription segments to diarization segments by timestamp overlap
    for trans_seg in transcribed_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        
        # Find overlapping diarization segment
        best_overlap = 0
        best_speaker = "Unknown"
        
        for diar_seg in diarization['segments']:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            # Calculate overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg['speaker']
        
        merged.append({
            **trans_seg,
            'speaker': best_speaker
        })
    
    print(f"✓ Merged {len(merged)} segments with speaker labels\n")
    
    return merged


def process_audio(audio_path, diarization_json=None, model_name='base'):
    """
    Complete timestamp-first transcription workflow
    Generates BOTH translated and original language versions
    
    Args:
        audio_path: Path to audio file
        diarization_json: Optional diarization JSON for speaker labels
        model_name: Whisper model size
    
    Returns:
        Tuple of (original_results, translated_results)
    """
    
    # Pass 1: Get timestamps (also generates translated version)
    timestamp_segments = get_timestamps_from_whisper(audio_path, model_name)
    
    # Save Pass 1 as translated version (Whisper translates everything to English)
    print(f"\n💡 Pass 1 generated TRANSLATED version (all English)\n")
    
    translated_segments = []
    for seg in timestamp_segments:
        translated_segments.append({
            'start': round(seg['start'], 3),
            'end': round(seg['end'], 3),
            'duration': round(seg['end'] - seg['start'], 3),
            'text': seg['text'].strip(),
            'language': 'en',
            'language_name': 'english',
            'note': 'auto-translated by Whisper'
        })
    
    # Pass 2: Re-transcribe each segment (preserves original languages)
    transcribed_segments, languages = retranscribe_segments_with_language(
        audio_path, 
        timestamp_segments, 
        model_name
    )
    
    print(f"\n💡 Pass 2 generated ORIGINAL version (preserved languages)\n")
    
    # Optional: Merge with diarization for both versions
    if diarization_json:
        original_final = merge_with_diarization(transcribed_segments, diarization_json)
        translated_final = merge_with_diarization(translated_segments, diarization_json)
    else:
        original_final = transcribed_segments
        translated_final = translated_segments
    
    # Build results for TRANSCRIPTION version (original languages)
    original_results = {
        'filename': Path(audio_path).name,
        'segments': original_final,
        'languages_detected': languages,
        'is_multilingual': len(languages) > 1,
        'model_used': model_name,
        'method': 'timestamp-first-retranscription',
        'version': 'transcription'  # Original transcription with preserved languages
    }
    
    # Build results for ENGLISH version (translated)
    translated_results = {
        'filename': Path(audio_path).name,
        'segments': translated_final,
        'languages_detected': ['en'],
        'is_multilingual': False,
        'model_used': model_name,
        'method': 'timestamp-first-translation',
        'version': 'english_version',  # All content translated to English
        'note': 'All segments translated to English by Whisper'
    }
    
    # Print summary
    print(f"{'='*60}")
    print(f"Transcription Complete - DUAL OUTPUT")
    print(f"{'='*60}")
    print(f"  File: {original_results['filename']}")
    print(f"\n  📄 TRANSCRIPTION (Original Languages):")
    print(f"    Segments: {len(original_final)}")
    print(f"    Languages: {', '.join(languages)}")
    print(f"    Multilingual: {original_results['is_multilingual']}")
    print(f"\n  🌐 ENGLISH VERSION (Translated):")
    print(f"    Segments: {len(translated_final)}")
    print(f"    Languages: English (all translated)")
    print(f"\nSample segments (TRANSCRIPTION):")
    for seg in original_final[:3]:
        speaker = f" ({seg['speaker']})" if 'speaker' in seg else ""
        print(f"  [{seg['start']:.1f}s]{speaker} [{seg['language']}] {seg['text'][:60]}...")
    print(f"\nSample segments (ENGLISH VERSION):")
    for seg in translated_final[:3]:
        speaker = f" ({seg['speaker']})" if 'speaker' in seg else ""
        print(f"  [{seg['start']:.1f}s]{speaker} [en] {seg['text'][:60]}...")
    print(f"{'='*60}\n")
    
    return original_results, translated_results


def save_results(results, output_path):
    """Save results to JSON"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {output_path}\n")


# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Timestamp-first transcription: accurate timestamps + non-translated text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python timestamp_transcription.py -f audio.wav
  
  # With diarization for speaker labels
  python timestamp_transcription.py -f audio.wav -d audio_diarization.json
  
  # Use better model
  python timestamp_transcription.py -f audio.wav -d audio_diarization.json --model medium
        """
    )
    
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Audio file to transcribe')
    parser.add_argument('-d', '--diarization', type=str,
                        help='Diarization JSON file for speaker labels (optional)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output JSON file (default: outputs/transcription/<audio>_timestamp_transcription.json)')

    parser.add_argument('--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    
    args = parser.parse_args()
    
    # Process - returns both versions
    original_results, translated_results = process_audio(args.file, args.diarization, args.model)
    
    # Save with organized structure
    if args.output:
        # User specified output - save original there
        original_path = args.output
        translated_path = Path(args.output).stem + '_translated.json'
    else:
        # Default: outputs/transcription/<filename>_*.json
        output_dir = Path('outputs/transcription')
        output_dir.mkdir(parents=True, exist_ok=True)
        original_path = output_dir / f"{Path(args.file).stem}_timestamp_transcription.json"
        translated_path = output_dir / f"{Path(args.file).stem}_timestamp_transcription_translated.json"
    
    # Save both versions
    save_results(original_results, original_path)
    save_results(translated_results, translated_path)
    
    print(f"✅ Generated TWO versions:")
    print(f"   📄 Original: {original_path}")
    print(f"   🌐 Translated: {translated_path}\n")

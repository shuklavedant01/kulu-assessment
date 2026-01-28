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
    
    # Transcribe to get timestamps (might translate, but timestamps are accurate)
    print(f"Processing for timestamp detection...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=True  # Show progress to ensure completion
    )
    
    print(f"✓ Found {len(result['segments'])} segments with timestamps")
    print(f"  Total duration: {result['segments'][-1]['end']:.1f}s\n")
    
    return result['segments']


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
    
    Args:
        audio_path: Path to audio file
        diarization_json: Optional diarization JSON for speaker labels
        model_name: Whisper model size
    
    Returns:
        Complete transcription results
    """
    
    # Pass 1: Get timestamps
    timestamp_segments = get_timestamps_from_whisper(audio_path, model_name)
    
    # Pass 2: Re-transcribe each segment
    transcribed_segments, languages = retranscribe_segments_with_language(
        audio_path, 
        timestamp_segments, 
        model_name
    )
    
    # Optional: Merge with diarization
    if diarization_json:
        final_segments = merge_with_diarization(transcribed_segments, diarization_json)
    else:
        final_segments = transcribed_segments
    
    # Build results
    results = {
        'filename': Path(audio_path).name,
        'segments': final_segments,
        'languages_detected': languages,
        'is_multilingual': len(languages) > 1,
        'model_used': model_name,
        'method': 'timestamp-first-retranscription'
    }
    
    # Print summary
    print(f"{'='*60}")
    print(f"Transcription Complete!")
    print(f"{'='*60}")
    print(f"  File: {results['filename']}")
    print(f"  Segments: {len(final_segments)}")
    print(f"  Languages: {', '.join(languages)}")
    print(f"  Multilingual: {results['is_multilingual']}")
    print(f"\nSample segments:")
    for seg in final_segments[:3]:
        speaker = f" ({seg['speaker']})" if 'speaker' in seg else ""
        print(f"  [{seg['start']:.1f}s]{speaker} [{seg['language']}] {seg['text'][:80]}...")
    print(f"{'='*60}\n")
    
    return results


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
    
    # Process
    results = process_audio(args.file, args.diarization, args.model)
    
    # Save with organized structure
    if args.output:
        output_path = args.output
    else:
        # Default: outputs/transcription/<filename>_timestamp_transcription.json
        output_dir = Path('outputs/transcription')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(args.file).stem}_timestamp_transcription.json"
    
    save_results(results, output_path)

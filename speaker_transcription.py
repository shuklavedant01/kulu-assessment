#!/usr/bin/env python3
"""
Speaker-Aware Transcription - Split by speaker then transcribe separately
This approach helps preserve original languages by transcribing each speaker independently
"""

import whisper
import json
from pathlib import Path
from pydub import AudioSegment
import argparse


def split_audio_by_speaker(audio_path, diarization_json):
    """
    Split audio file into separate chunks per speaker
    
    Args:
        audio_path: Path to audio file
        diarization_json: Path to diarization JSON with segments
    
    Returns:
        Dictionary with speaker-separated audio chunks
    """
    
    print(f"\nSplitting audio by speaker...")
    
    # Load original audio
    audio = AudioSegment.from_wav(str(audio_path))
    
    # Load diarization results
    with open(diarization_json, 'r', encoding='utf-8') as f:
        diarization = json.load(f)
    
    # Group segments by speaker
    speaker_chunks = {}
    
    for segment in diarization['segments']:
        speaker = segment['speaker']
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        
        # Extract audio chunk
        chunk = audio[start_ms:end_ms]
        
        # Add to speaker's collection
        if speaker not in speaker_chunks:
            speaker_chunks[speaker] = {
                'chunks': [],
                'timestamps': [],
                'total_duration': 0
            }
        
        speaker_chunks[speaker]['chunks'].append(chunk)
        speaker_chunks[speaker]['timestamps'].append({
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['duration']
        })
        speaker_chunks[speaker]['total_duration'] += segment['duration']
    
    print(f"  Found {len(speaker_chunks)} speakers:")
    for speaker, data in speaker_chunks.items():
        print(f"    {speaker}: {len(data['chunks'])} segments, {data['total_duration']:.1f}s total")
    
    return speaker_chunks


def transcribe_speaker_chunks(speaker_chunks, model_name='base', language_hints=None):
    """
    Transcribe audio chunks for each speaker separately
    
    Args:
        speaker_chunks: Dictionary of speaker audio chunks
        model_name: Whisper model size
        language_hints: Dictionary mapping speaker to language code (e.g., {'Agent': 'ru', 'User': 'en'})
    
    Returns:
        Dictionary with transcriptions per speaker
    """
    
    print(f"\nTranscribing speakers separately...")
    print(f"Loading Whisper '{model_name}' model...")
    
    # Load Whisper model
    model = whisper.load_model(model_name)
    
    results = {}
    
    for speaker, data in speaker_chunks.items():
        print(f"\n  Transcribing {speaker}...")
        
        # Combine all chunks for this speaker into one audio file
        combined_audio = data['chunks'][0]
        for chunk in data['chunks'][1:]:
            combined_audio += chunk
        
        # Save temporary file
        temp_file = f"temp_{speaker}.wav"
        combined_audio.export(temp_file, format='wav')
        
        # Get language hint for this speaker
        language = language_hints.get(speaker) if language_hints else None
        
        # Transcribe
        result = model.transcribe(
            temp_file,
            language=language,
            task='transcribe',
            word_timestamps=True,
            verbose=False
        )
        
        detected_lang = result.get('language', 'unknown')
        lang_name = whisper.tokenizer.LANGUAGES.get(detected_lang, detected_lang)
        
        print(f"    Language: {lang_name}")
        print(f"    Text: {result['text'][:100]}...")
        
        # Store results
        results[speaker] = {
            'text': result['text'].strip(),
            'language': detected_lang,
            'language_name': lang_name,
            'segments': result['segments'],
            'timestamps': data['timestamps'],
            'chunk_count': len(data['chunks'])
        }
        
        # Clean up temp file
        Path(temp_file).unlink()
    
    return results


def merge_transcriptions(speaker_transcriptions, original_segments):
    """
    Merge speaker transcriptions back into timeline order
    
    Args:
        speaker_transcriptions: Transcription results per speaker
        original_segments: Original diarization segments with timestamps
    
    Returns:
        Merged segments with speaker labels and transcriptions
    """
    
    print(f"\nMerging transcriptions into timeline...")
    
    merged_segments = []
    languages_detected = set()
    
    # Map each original segment to its transcription
    for orig_seg in original_segments:
        speaker = orig_seg['speaker']
        
        if speaker in speaker_transcriptions:
            trans = speaker_transcriptions[speaker]
            languages_detected.add(trans['language'])
            
            # Find matching timestamp in speaker's transcription
            # For now, we'll assign the full speaker text proportionally
            # (More sophisticated alignment would use word timestamps)
            
            merged_segments.append({
                'start': orig_seg['start'],
                'end': orig_seg['end'],
                'duration': orig_seg['duration'],
                'speaker': speaker,
                'language': trans['language'],
                'text': f"[{speaker} - {trans['language_name']}] (See full transcription per speaker)"
            })
    
    is_multilingual = len(languages_detected) > 1
    
    return {
        'segments': merged_segments,
        'languages_detected': sorted(list(languages_detected)),
        'is_multilingual': is_multilingual,
        'speaker_transcriptions': speaker_transcriptions
    }


def process_audio_with_diarization(audio_path, diarization_json, model_name='base', language_hints=None):
    """
    Complete workflow: split by speaker and transcribe
    
    Args:
        audio_path: Path to audio file
        diarization_json: Path to diarization JSON
        model_name: Whisper model
        language_hints: Language hints per speaker
    
    Returns:
        Complete transcription results
    """
    
    print(f"{'='*60}")
    print(f"Speaker-Aware Transcription")
    print(f"{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Diarization: {diarization_json}")
    
    # Step 1: Split audio by speaker
    speaker_chunks = split_audio_by_speaker(audio_path, diarization_json)
    
    # Step 2: Transcribe each speaker
    speaker_transcriptions = transcribe_speaker_chunks(speaker_chunks, model_name, language_hints)
    
    # Step 3: Load original segments for timeline
    with open(diarization_json, 'r', encoding='utf-8') as f:
        diarization = json.load(f)
    
    # Step 4: Merge results
    results = merge_transcriptions(speaker_transcriptions, diarization['segments'])
    
    # Add metadata
    results['filename'] = Path(audio_path).name
    results['model_used'] = model_name
    
    print(f"\n{'='*60}")
    print(f"Transcription Complete!")
    print(f"  Languages: {', '.join(results['languages_detected'])}")
    print(f"  Multilingual: {results['is_multilingual']}")
    print(f"\nSpeaker Transcriptions:")
    for speaker, trans in results['speaker_transcriptions'].items():
        print(f"\n  {speaker} ({trans['language_name']}):")
        print(f"    {trans['text'][:200]}...")
    print(f"{'='*60}\n")
    
    return results


def save_results(results, output_path):
    """Save transcription results to JSON"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {output_path}\n")


# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe audio by splitting speakers first (preserves multilingual content)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect languages)
  python speaker_transcription.py -f audio.wav -d audio_diarization.json
  
  # Specify language hints per speaker
  python speaker_transcription.py -f audio.wav -d audio_diarization.json --agent-lang ru --user-lang en
  
  # Use better model
  python speaker_transcription.py -f audio.wav -d audio_diarization.json --model medium
        """
    )
    
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Audio file to transcribe')
    parser.add_argument('-d', '--diarization', type=str, required=True,
                        help='Diarization JSON file with speaker segments')
    parser.add_argument('-o', '--output', type=str,
                        help='Output JSON file (default: <audio>_speaker_transcription.json)')
    parser.add_argument('--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('--agent-lang', type=str,
                        help='Force language for Agent (e.g., ru, en, es)')
    parser.add_argument('--user-lang', type=str,
                        help='Force language for User (e.g., en, ru, es)')
    
    args = parser.parse_args()
    
    # Build language hints
    language_hints = {}
    if args.agent_lang:
        language_hints['Agent'] = args.agent_lang
    if args.user_lang:
        language_hints['User'] = args.user_lang
    
    # Process audio
    results = process_audio_with_diarization(
        args.file,
        args.diarization,
        args.model,
        language_hints if language_hints else None
    )
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.file).stem + '_speaker_transcription.json'
    
    save_results(results, output_path)

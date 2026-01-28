#!/usr/bin/env python3
"""
Transcription - Speech-to-text with language detection using Whisper
"""

# Import required libraries
import whisper
import json
from pathlib import Path
import argparse


def transcribe_audio(audio_path, model_name='base', language=None):
    """
    Transcribe audio file using OpenAI Whisper
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Force specific language (None = auto-detect)
    
    Returns:
        Dictionary with transcription results and metadata
    """
    
    print(f"\nTranscribing: {audio_path}")
    print(f"Loading Whisper '{model_name}' model...")
    
    # Load Whisper model
    model = whisper.load_model(model_name)
    
    # Transcribe audio
    print(f"Processing audio...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,  # Enable word-level timestamps
        verbose=False
    )
    
    # Extract language information
    detected_language = result.get('language', 'unknown')
    language_name = whisper.tokenizer.LANGUAGES.get(detected_language, detected_language)
    
    print(f"  Language detected: {language_name} ({detected_language})")
    print(f"  Transcription complete!")
    
    # Process segments with timestamps
    segments = []
    for segment in result['segments']:
        segments.append({
            'start': round(segment['start'], 3),
            'end': round(segment['end'], 3),
            'duration': round(segment['end'] - segment['start'], 3),
            'text': segment['text'].strip(),
            'language': detected_language
        })
    
    # Collect all unique languages (for code-switching detection)
    languages_used = set()
    if 'segments' in result:
        for segment in result['segments']:
            if 'language' in segment:
                languages_used.add(segment.get('language', detected_language))
    if not languages_used:
        languages_used.add(detected_language)
    
    # Check if multiple languages were used
    is_multilingual = len(languages_used) > 1
    
    return {
        'filename': Path(audio_path).name,
        'text': result['text'].strip(),
        'language': detected_language,
        'language_name': language_name,
        'languages_detected': sorted(list(languages_used)),
        'is_multilingual': is_multilingual,
        'segments': segments,
        'model_used': model_name
    }


def detect_languages(audio_path, model_name='base'):
    """
    Quick language detection without full transcription
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
    
    Returns:
        List of detected languages and whether it's multilingual
    """
    
    print(f"\nDetecting languages in: {audio_path}")
    
    # Load model
    model = whisper.load_model(model_name)
    
    # Transcribe with language detection
    result = model.transcribe(str(audio_path), verbose=False)
    
    # Get detected language
    languages = [result.get('language', 'unknown')]
    
    # Check segments for multiple languages
    if 'segments' in result:
        segment_langs = set()
        for segment in result['segments']:
            if 'language' in segment:
                segment_langs.add(segment['language'])
        if segment_langs:
            languages = sorted(list(segment_langs))
    
    is_multilingual = len(languages) > 1
    
    print(f"  Languages: {', '.join(languages)}")
    print(f"  Multilingual: {is_multilingual}")
    
    return languages, is_multilingual


def save_transcription(results, output_folder, audio_filename):
    """Save transcription results to JSON file"""
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = output_path / f"{Path(audio_filename).stem}_transcription.json"
    
    # Write JSON with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {output_file.name}\n")
    return str(output_file)


def process_folder(input_folder='outputs/converted', output_folder='outputs/transcription', model_name='base'):
    """
    Process all WAV files for transcription
    
    Args:
        input_folder: Input folder with WAV files
        output_folder: Output folder for JSON results
        model_name: Whisper model size
    """
    
    # Get all WAV files from input folder
    input_path = Path(input_folder)
    audio_files = list(input_path.glob('*.wav'))
    
    if not audio_files:
        print(f"No WAV files found in {input_folder}")
        return []
    
    print(f"{'='*60}")
    print(f"Whisper Transcription")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Found {len(audio_files)} WAV file(s) in {input_folder}\n")
    
    # Process each audio file
    all_results = []
    for audio_file in audio_files:
        try:
            # Transcribe
            results = transcribe_audio(audio_file, model_name)
            
            # Save to JSON
            save_transcription(results, output_folder, audio_file.name)
            
            # Print summary
            print(f"  Summary:")
            print(f"    Language: {results['language_name']}")
            print(f"    Multilingual: {results['is_multilingual']}")
            print(f"    Segments: {len(results['segments'])}")
            print(f"    Text length: {len(results['text'])} chars\n")
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {e}\n")
    
    # Print final summary
    print(f"{'='*60}")
    print(f"Transcription complete! Processed {len(all_results)}/{len(audio_files)} file(s)")
    print(f"Results saved to: {output_folder}")
    print(f"{'='*60}\n")
    
    return all_results


# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe audio files using OpenAI Whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe all WAV files (base model)
  python transcription.py
  
  # Transcribe single file
  python transcription.py -f outputs/converted/audio1.wav
  
  # Use different model size
  python transcription.py -f audio.wav --model small
  
  # Process folder with custom output
  python transcription.py -i my_audio -o my_transcripts --model base
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default='outputs/converted',
                        help='Input folder with WAV files (default: outputs/converted)')
    parser.add_argument('-o', '--output', type=str, default='outputs/transcription',
                        help='Output folder for JSON results (default: outputs/transcription)')
    parser.add_argument('-f', '--file', type=str,
                        help='Process single audio file instead of folder')
    parser.add_argument('--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('--language', type=str,
                        help='Force specific language (e.g., en, ru, es)')
    
    args = parser.parse_args()
    
    # Process single file or entire folder
    if args.file:
        # Process single file
        audio_file = Path(args.file)
        results = transcribe_audio(audio_file, args.model, args.language)
        save_transcription(results, args.output, args.file)
        
        # Print results
        print(f"\nTranscription Results:")
        print(f"  Language: {results['language_name']}")
        print(f"  Multilingual: {results['is_multilingual']}")
        print(f"  Text: {results['text'][:200]}...")  # First 200 chars
    else:
        # Process entire folder
        process_folder(args.input, args.output, args.model)

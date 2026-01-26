#!/usr/bin/env python3
"""
Audio Converter - Converts audio files to 16kHz WAV mono format
"""

# Import required libraries for file handling, CLI, and audio processing
import argparse
from pathlib import Path
from pydub import AudioSegment

# Target format specifications
TARGET_RATE = 16000  # 16kHz sample rate
TARGET_CHANNELS = 1  # Mono audio


# Main conversion function
def convert_audio_files(input_folder='audio_files', output_folder='outputs/converted'):
    """Convert all audio files in input folder to 16kHz WAV mono"""
    
    # Setup input and output paths
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files from input folder
    audio_files = [f for f in input_path.glob('*') if f.suffix.lower() in 
                   ['.mp3', '.wav', '.ogg', '.oga', '.flac', '.m4a', '.aac']]
    
    print(f"Found {len(audio_files)} audio file(s)")
    print(f"Converting to: {TARGET_RATE}Hz, {TARGET_CHANNELS} channel, WAV\n")
    
    # Process each audio file
    for audio_file in audio_files:
        try:
            print(f"Processing: {audio_file.name}")
            
            # Load the audio file
            audio = AudioSegment.from_file(str(audio_file))
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(TARGET_CHANNELS)
            
            # Resample to target rate
            audio = audio.set_frame_rate(TARGET_RATE)
            
            # Save as WAV with same name
            output_file = output_path / f"{audio_file.stem}.wav"
            audio.export(str(output_file), format='wav')
            
            print(f"  ✓ Saved: {output_file.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print(f"Conversion complete! Files saved to: {output_folder}")


# CLI argument parsing
if __name__ == '__main__':
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Convert audio to 16kHz WAV mono')
    parser.add_argument('-i', '--input', default='audio_files', 
                        help='Input folder (default: audio_files)')
    parser.add_argument('-o', '--output', default='outputs/converted',
                        help='Output folder (default: outputs/converted)')
    
    # Parse arguments and run conversion
    args = parser.parse_args()
    convert_audio_files(args.input, args.output)

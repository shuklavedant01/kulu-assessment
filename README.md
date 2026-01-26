# Audio File Converter

A CLI tool for converting audio files to a standardized format (16kHz WAV mono) for consistent audio processing.

## Features

- ✅ Converts multiple audio formats to 16kHz WAV mono
- ✅ Batch processing of entire folders
- ✅ Support for 10+ audio formats (MP3, OGG, FLAC, M4A, etc.)
- ✅ Detailed logging and progress reporting
- ✅ Command-line interface for easy automation
- ✅ Configurable input/output folders

## Requirements

- Python 3.7+
- FFmpeg (required by pydub for audio conversion)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

**Windows:**
- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add FFmpeg to your system PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Usage

### Basic Usage

Convert all audio files in the `audio_files` folder:

```bash
python audio_converter.py
```

This will:
- Read audio files from `audio_files/` folder
- Convert them to 16kHz WAV mono format
- Save results to `outputs/converted/` folder

### Custom Folders

Specify custom input and output folders:

```bash
# Custom input folder only
python audio_converter.py --input my_audio

# Custom input and output folders
python audio_converter.py --input raw_audio --output processed
```

### With Verbose Logging

```bash
python audio_converter.py --verbose
```

### Help

```bash
python audio_converter.py --help
```

## Supported Formats

The tool supports the following input audio formats:
- MP3
- WAV
- OGG/OGA
- FLAC
- M4A
- AAC
- WMA
- AIFF
- OPUS
- WebM

## Output Format

All files are converted to:
- **Sample Rate:** 16,000 Hz (16 kHz)
- **Channels:** 1 (Mono)
- **Format:** WAV (uncompressed)

## Configuration

You can modify default settings in `config.yaml`:

```yaml
audio:
  input_folder: "audio_files"
  output_folder: "outputs/converted"
  target_format:
    sample_rate: 16000
    channels: 1
    format: "wav"
```

## Project Structure

```
kulu-assessment/
├── audio_files/          # Input audio files
│   ├── audio1.oga
│   ├── audio2.oga
│   └── audio3.oga
├── outputs/
│   └── converted/        # Converted output files
├── audio_converter.py    # Main conversion script
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Examples

### Example 1: Convert all files in audio_files folder

```bash
python audio_converter.py
```

Output:
```
======================================================================
Audio Converter - Convert to 16kHz WAV Mono
======================================================================

2026-01-27 00:49:30 - INFO - Found 3 audio file(s) in audio_files

Starting conversion of 3 file(s)...
Target format: 16000Hz, 1 channel(s), WAV
----------------------------------------------------------------------
2026-01-27 00:49:30 - INFO - Processing: audio1.oga
2026-01-27 00:49:30 - INFO -   Original: 48000Hz, 2 channel(s), 42.15s
2026-01-27 00:49:31 - INFO -   Converted: 16000Hz, 1 channel(s)
2026-01-27 00:49:31 - INFO -   Saved to: audio1.wav (1.35 MB)
----------------------------------------------------------------------

Conversion complete!
Successful: 3
Failed: 0
Output folder: outputs/converted
```

### Example 2: Custom folders

```bash
python audio_converter.py --input recordings --output processed_audio
```

## Troubleshooting

### Error: "FFmpeg not found"

Install FFmpeg and ensure it's in your system PATH. See Installation section above.

### Error: "No audio files found"

- Check that the input folder exists
- Verify that the folder contains supported audio formats
- Use `--verbose` flag for detailed logging

### Permission Errors

Ensure you have read permissions for the input folder and write permissions for the output folder.

## License

MIT License

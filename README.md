# Audio Processing Tools

CLI tools for processing, analyzing, and visualizing audio files for conversation analysis.

## Features

✅ **Audio Conversion** - Convert to 16kHz WAV mono format  
✅ **Audio Analysis** - Check duration, sample rate, channels, bit depth  
✅ **Quality Verification** - Detect corrupted files and verify content  
✅ **Waveform Visualization** - Generate conversation structure plots  
✅ **Complete Pipeline** - Run all steps together with one command

## Requirements

- Python 3.7+
- FFmpeg (required by pydub)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

**Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to system PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Process everything in one command:

```bash
python main.py
```

This will:
1. Convert audio files to 16kHz WAV mono
2. Analyze audio properties and quality
3. Generate waveform visualizations

**Options:**
```bash
# Skip conversion (analyze existing files)
python main.py --no-convert

# Only visualize
python main.py --no-convert --no-analyze

# Custom input folder
python main.py -i my_audio_files
```

### Option 2: Run Individual Tools

#### Audio Converter

Convert files to 16kHz WAV mono:

```bash
# Convert all files in audio_files/
python audio_converter.py

# Custom folders
python audio_converter.py -i input_folder -o output_folder
```

#### Audio Analyzer

Check audio properties and verify quality:

```bash
# Analyze all files in audio_files/
python analyzer.py

# Analyze specific file
python analyzer.py -f audio_files/audio1.oga

# Custom folder
python analyzer.py -d my_audio_folder
```

**Output includes:**
- Duration (seconds/minutes)
- Sample rate (Hz)
- Channels (mono/stereo)
- Bit depth
- File size
- Corruption check
- Content verification
- Speech/sound segments detected

#### Waveform Visualizer

Generate waveform plots:

```bash
# Visualize all files in audio_files/
python visualizer.py

# Visualize specific file
python visualizer.py -f audio_files/audio1.oga

# Custom folders
python visualizer.py -d input_folder -o output_folder
```

**Output:**
- Dual-plot visualization saved as PNG
- Full waveform (shows amplitude over time)
- Envelope plot (shows conversation structure)

## Examples

### Analyze Audio Files

```bash
python analyzer.py
```

Output:
```
Found 3 audio file(s) in audio_files

Analyzing: audio1.oga...

============================================================
File: audio1.oga
============================================================

📊 Audio Properties:
  Duration:     42.15 seconds (0.70 minutes)
  Sample Rate:  48000 Hz
  Channels:     2 (Stereo)
  Bit Depth:    16 bits
  File Size:    1.62 MB

✓ Quality Check:
  Corrupted:    ✅ No
  Has Content:  ✅ Yes
  Content:      95.3% of file
  Speech/Sound: 12 segments detected
```

### Generate Visualizations

```bash
python visualizer.py
```

Creates waveform plots in `outputs/visualizations/` showing:
- Audio amplitude over time
- Conversation energy envelope
- File metadata

## Project Structure

```
kulu-assessment/
├── audio_files/              # Input audio files
│   ├── audio1.oga
│   ├── audio2.oga
│   └── audio3.oga
├── outputs/
│   ├── converted/           # Converted WAV files
│   └── visualizations/      # Waveform PNG images
├── audio_converter.py       # Audio conversion tool
├── analyzer.py              # Audio analysis tool
├── visualizer.py            # Waveform visualization tool
├── main.py                  # Complete pipeline
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## Supported Audio Formats

MP3, WAV, OGG, OGA, FLAC, M4A, AAC

## Output Files

- **Converted Audio:** `outputs/converted/*.wav` (16kHz, mono)
- **Visualizations:** `outputs/visualizations/*_waveform.png`

## Troubleshooting

### FFmpeg not found
Install FFmpeg and add to system PATH

### Import errors
Run `pip install -r requirements.txt`

### No audio files found
Check input folder path and file formats

## License

MIT License

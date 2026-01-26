#!/usr/bin/env python3
"""
Audio Visualizer - Create waveform visualizations
"""

# Import required libraries for visualization
import argparse
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Configure matplotlib for better-looking plots
matplotlib.use('Agg')  # Use non-interactive backend for saving files
plt.style.use('seaborn-v0_8-darkgrid')


# Function to create waveform visualization
def visualize_waveform(audio_path, output_folder='outputs/visualizations'):
    """Generate waveform visualization for audio file"""
    
    # Setup output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizing: {audio_path.name}...")
    
    # Load audio file
    audio = AudioSegment.from_file(str(audio_path))
    
    # Convert to numpy array (samples)
    samples = np.array(audio.get_array_of_samples())
    
    # If stereo, convert to mono by averaging channels
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)
    
    # Create time axis in seconds
    duration = len(audio) / 1000.0
    time_axis = np.linspace(0, duration, len(samples))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Full waveform
    ax1.plot(time_axis, samples, linewidth=0.5, color='#2E86AB', alpha=0.8)
    ax1.set_title(f'Waveform: {audio_path.name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)
    
    # Plot 2: Envelope (showing conversation structure)
    # Calculate envelope using absolute values with smoothing
    window_size = len(samples) // 1000 or 1  # Adjust for smoothing
    envelope = np.abs(samples)
    envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
    
    # Downsample for clearer visualization
    downsample_factor = max(1, len(samples) // 10000)
    time_downsampled = time_axis[::downsample_factor]
    envelope_downsampled = envelope[::downsample_factor]
    
    ax2.fill_between(time_downsampled, envelope_downsampled, alpha=0.6, color='#A23B72')
    ax2.plot(time_downsampled, envelope_downsampled, linewidth=1, color='#F18F01', alpha=0.9)
    ax2.set_title('Conversation Structure (Envelope)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Energy', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, duration)
    
    # Add metadata text
    metadata = f"Duration: {duration:.2f}s | Sample Rate: {audio.frame_rate}Hz | Channels: {audio.channels}"
    fig.text(0.5, 0.02, metadata, ha='center', fontsize=10, style='italic', color='gray')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_file = output_path / f"{audio_path.stem}_waveform.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}\n")
    return str(output_file)


# Function to visualize multiple files
def visualize_folder(folder_path, output_folder='outputs/visualizations'):
    """Visualize all audio files in a folder"""
    
    # Get all audio files
    audio_path = Path(folder_path)
    audio_files = [f for f in audio_path.glob('*') if f.suffix.lower() in 
                   ['.mp3', '.wav', '.ogg', '.oga', '.flac', '.m4a', '.aac']]
    
    print(f"Found {len(audio_files)} audio file(s)\n")
    
    # Visualize each file
    output_files = []
    for audio_file in audio_files:
        try:
            output_file = visualize_waveform(audio_file, output_folder)
            output_files.append(output_file)
        except Exception as e:
            print(f"❌ Error visualizing {audio_file.name}: {e}\n")
    
    # Print summary
    print(f"Visualization complete! {len(output_files)} file(s) saved to: {output_folder}")
    return output_files


# CLI argument parsing
if __name__ == '__main__':
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Visualize audio waveforms')
    parser.add_argument('-f', '--file', type=str,
                        help='Single audio file to visualize')
    parser.add_argument('-d', '--dir', type=str, default='audio_files',
                        help='Directory containing audio files (default: audio_files)')
    parser.add_argument('-o', '--output', type=str, default='outputs/visualizations',
                        help='Output folder for visualizations (default: outputs/visualizations)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Visualize single file or entire directory
    if args.file:
        visualize_waveform(Path(args.file), args.output)
    else:
        visualize_folder(args.dir, args.output)

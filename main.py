#!/usr/bin/env python3
"""
Main Pipeline - Run audio conversion, analysis, and visualization
"""

# Import modules
import argparse
from pathlib import Path
from audio_converter import convert_audio_files
from analyzer import analyze_folder
from visualizer import visualize_folder


# Main pipeline function
def run_pipeline(input_folder='audio_files', convert=True, analyze=True, visualize=True):
    """Run complete audio processing pipeline"""
    
    print("\n" + "="*70)
    print("AUDIO PROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Convert audio files (optional)
    if convert:
        print("STEP 1: Converting audio files to 16kHz WAV mono")
        print("-"*70)
        convert_audio_files(input_folder, 'outputs/converted')
        print()
    
    # Step 2: Analyze audio files
    if analyze:
        print("STEP 2: Analyzing audio properties and quality")
        print("-"*70)
        # Analyze converted files if we converted, otherwise analyze originals
        analyze_path = 'outputs/converted' if convert else input_folder
        analyze_folder(analyze_path)
        print()
    
    # Step 3: Visualize waveforms
    if visualize:
        print("STEP 3: Generating waveform visualizations")
        print("-"*70)
        # Visualize converted files if we converted, otherwise visualize originals
        viz_path = 'outputs/converted' if convert else input_folder
        visualize_folder(viz_path, 'outputs/visualizations')
        print()
    
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70 + "\n")


# CLI argument parsing
if __name__ == '__main__':
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description='Audio processing pipeline: convert, analyze, and visualize',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py
  
  # Only analyze (skip conversion)
  python main.py --no-convert
  
  # Only visualize (skip conversion and analysis)
  python main.py --no-convert --no-analyze
  
  # Specify custom input folder
  python main.py -i my_audio_files
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default='audio_files',
                        help='Input folder (default: audio_files)')
    parser.add_argument('--no-convert', action='store_true',
                        help='Skip audio conversion step')
    parser.add_argument('--no-analyze', action='store_true',
                        help='Skip audio analysis step')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip waveform visualization step')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run pipeline with specified options
    run_pipeline(
        input_folder=args.input,
        convert=not args.no_convert,
        analyze=not args.no_analyze,
        visualize=not args.no_visualize
    )

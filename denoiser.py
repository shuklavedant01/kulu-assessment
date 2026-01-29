#!/usr/bin/env python3
"""
Advanced Denoising using Demucs
Separates vocals from background noise/music using deep learning U-Net
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import torch

def denoise_audio(input_file, output_file, model='htdemucs'):
    """
    Denoise audio using Demucs (Keep only vocals stem)
    
    Args:
        input_file: Path to input audio
        output_file: Path to save enhanced audio
        model: Demucs model version (default: htdemucs)
    """
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🧹 Denoising: {input_path.name}")
    print(f"   Model: {model} (Separating vocals)")
    
    # Demucs command
    # -n: model name
    # --two-stems=vocals: ONLY separate vocals and "other" (faster)
    # -o: output directory
    cmd = [
        'demucs',
        '-n', model,
        '--two-stems', 'vocals',  # Optimizes for speech/vocals
        str(input_path),
        '-o', str(output_path.parent / 'temp_demucs')
    ]
    
    try:
        # Run Demucs
        print("   Running Demucs isolation... (this may take a moment)")
        subprocess.run(cmd, check=True)
        
        # Locate the output file
        # Demucs structure: <out>/<model>/<track_name>/vocals.wav
        source_name = input_path.stem
        # Demucs output folder name matches inputs filename usually
        demucs_out_dir = output_path.parent / 'temp_demucs' / model / source_name
        vocals_file = demucs_out_dir / 'vocals.wav'
        
        if not vocals_file.exists():
            # Sometimes demucs changes naming, check folder content
            found = list((output_path.parent / 'temp_demucs' / model).glob('**/vocals.wav'))
            if found:
                vocals_file = found[0]
            else:
                raise FileNotFoundError(f"Demucs failed to produce output at {vocals_file}")
            
        # Move/Rename to final location
        shutil.move(str(vocals_file), str(output_path))
        print(f"   ✓ Vocals extracted to: {output_path.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Demucs Error: {e}")
        raise
    except FileNotFoundError:
        print("❌ Demucs command not found. Please install: pip install demucs")
        raise
    finally:
        # Cleanup temp folder
        temp_dir = output_path.parent / 'temp_demucs'
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    return str(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demucs Audio Denoiser (Vocals Isolation)')
    parser.add_argument('-i', '--input', required=True, help='Input audio file')
    parser.add_argument('-o', '--output', required=True, help='Output enhanced audio file')
    parser.add_argument('--model', default='htdemucs', help='Demucs model (htdemucs, htdemucs_ft, mdx_extra_q)')
    
    args = parser.parse_args()
    
    denoise_audio(args.input, args.output, args.model)

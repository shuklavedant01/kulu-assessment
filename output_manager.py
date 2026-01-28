#!/usr/bin/env python3
"""
Output Manager - Organize all processing results in structured folders
"""

from pathlib import Path
import json
import shutil
from datetime import datetime


class OutputManager:
    """Manages organized output structure for audio processing pipeline"""
    
    def __init__(self, base_dir='outputs'):
        """
        Initialize output manager
        
        Args:
            base_dir: Base directory for all outputs (default: 'outputs')
        """
        self.base_dir = Path(base_dir)
        self.structure = {
            'converted': self.base_dir / 'converted',      # WAV files from conversion
            'diarization': self.base_dir / 'diarization',  # Speaker diarization JSON
            'transcription': self.base_dir / 'transcription',  # Whisper transcription JSON
            'final': self.base_dir / 'final',              # Merged final results
            'logs': self.base_dir / 'logs',                # Processing logs
            'temp': self.base_dir / 'temp'                 # Temporary files
        }
        
        # Create all directories
        self._create_structure()
    
    def _create_structure(self):
        """Create all output directories"""
        for folder in self.structure.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, category, filename):
        """
        Get full path for a file in specific category
        
        Args:
            category: One of 'converted', 'diarization', 'transcription', 'final', 'logs', 'temp'
            filename: Name of the file
        
        Returns:
            Full path to file
        """
        if category not in self.structure:
            raise ValueError(f"Unknown category: {category}. Use: {list(self.structure.keys())}")
        
        return self.structure[category] / filename
    
    def save_json(self, data, category, filename):
        """
        Save data as JSON to specified category
        
        Args:
            data: Dictionary to save
            category: Output category
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.get_path(category, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def load_json(self, category, filename):
        """
        Load JSON from specified category
        
        Args:
            category: Category folder
            filename: Filename to load
        
        Returns:
            Loaded data
        """
        file_path = self.get_path(category, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_audio_results(self, audio_name):
        """
        Get all results for a specific audio file
        
        Args:
            audio_name: Name of audio file (without extension)
        
        Returns:
            Dictionary with paths to all results
        """
        results = {
            'audio': self.get_path('converted', f'{audio_name}.wav'),
            'diarization': self.get_path('diarization', f'{audio_name}_diarization.json'),
            'transcription': self.get_path('transcription', f'{audio_name}_timestamp_transcription.json'),
            'final': self.get_path('final', f'{audio_name}_complete.json')
        }
        
        # Check which files exist
        for key, path in results.items():
            results[f'{key}_exists'] = path.exists()
        
        return results
    
    def create_summary(self):
        """
        Create summary of all processed files
        
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'base_directory': str(self.base_dir.absolute()),
            'categories': {}
        }
        
        for category, folder in self.structure.items():
            files = []
            if folder.exists():
                if category in ['converted']:
                    files = list(folder.glob('*.wav'))
                elif category in ['diarization', 'transcription', 'final']:
                    files = list(folder.glob('*.json'))
                elif category == 'logs':
                    files = list(folder.glob('*.log'))
            
            summary['categories'][category] = {
                'path': str(folder),
                'file_count': len(files),
                'files': [f.name for f in files]
            }
        
        return summary
    
    def print_structure(self):
        """Print organized directory structure"""
        print(f"\n{'='*60}")
        print(f"Output Directory Structure")
        print(f"{'='*60}")
        print(f"Base: {self.base_dir.absolute()}\n")
        
        summary = self.create_summary()
        
        for category, info in summary['categories'].items():
            print(f"📁 {category}/ ({info['file_count']} files)")
            for filename in info['files'][:5]:  # Show first 5
                print(f"   └─ {filename}")
            if info['file_count'] > 5:
                print(f"   └─ ... and {info['file_count'] - 5} more")
            print()
        
        print(f"{'='*60}\n")
    
    def cleanup_temp(self):
        """Remove all temporary files"""
        temp_dir = self.structure['temp']
        if temp_dir.exists():
            for f in temp_dir.glob('*'):
                f.unlink()
            print(f"✓ Cleaned up temporary files")


# CLI for testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage output directory structure')
    parser.add_argument('--base', type=str, default='outputs',
                        help='Base output directory (default: outputs)')
    parser.add_argument('--show', action='store_true',
                        help='Show current structure')
    parser.add_argument('--summary', action='store_true',
                        help='Create summary JSON')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up temporary files')
    
    args = parser.parse_args()
    
    # Initialize output manager
    manager = OutputManager(args.base)
    
    if args.show:
        manager.print_structure()
    
    if args.summary:
        summary = manager.create_summary()
        output_path = manager.save_json(summary, 'logs', 'processing_summary.json')
        print(f"✓ Summary saved to: {output_path}")
    
    if args.cleanup:
        manager.cleanup_temp()
    
    if not (args.show or args.summary or args.cleanup):
        # Default: show structure
        manager.print_structure()

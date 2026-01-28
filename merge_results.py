#!/usr/bin/env python3
"""
Merge Transcription + Diarization Results
Combines Whisper transcription with speaker diarization using timestamp overlap
"""

import json
from pathlib import Path
import argparse


def calculate_overlap(start1, end1, start2, end2):
    """
    Calculate overlap duration between two time segments
    
    Args:
        start1, end1: First segment
        start2, end2: Second segment
    
    Returns:
        Overlap duration in seconds
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def find_speaker_for_segment(trans_seg, diarization_segments, min_overlap_ratio=0.3):
    """
    Find the speaker for a transcription segment using overlap matching
    Fallback to nearest speaker if no good overlap found
    
    Args:
        trans_seg: Transcription segment with start/end
        diarization_segments: List of diarization segments
        min_overlap_ratio: Minimum overlap ratio required (default: 30%)
    
    Returns:
        Speaker label or "Unknown"
    """
    
    trans_start = trans_seg['start']
    trans_end = trans_seg['end']
    trans_duration = trans_seg['duration']
    
    # Skip empty segments
    if trans_duration == 0:
        return None
    
    # Find all overlapping diarization segments
    overlaps = []
    
    for diar_seg in diarization_segments:
        overlap = calculate_overlap(
            trans_start, trans_end,
            diar_seg['start'], diar_seg['end']
        )
        
        if overlap > 0:
            overlaps.append({
                'speaker': diar_seg['speaker'],
                'overlap': overlap,
                'overlap_ratio': overlap / trans_duration
            })
    
    # If we have overlaps, use the best one
    if overlaps:
        best_match = max(overlaps, key=lambda x: x['overlap'])
        return best_match['speaker']
    
    # No overlaps found - find nearest speaker (fallback for gaps)
    nearest_speaker = None
    min_distance = float('inf')
    
    for diar_seg in diarization_segments:
        # Calculate distance to this segment
        if trans_end < diar_seg['start']:
            distance = diar_seg['start'] - trans_end
        elif trans_start > diar_seg['end']:
            distance = trans_start - diar_seg['end']
        else:
            distance = 0
        
        if distance < min_distance:
            min_distance = distance
            nearest_speaker = diar_seg['speaker']
    
    # Use nearest speaker if within 2 seconds
    if nearest_speaker and min_distance < 2.0:
        return nearest_speaker
    
    return "Unknown"


def merge_transcription_diarization(transcription_json, diarization_json, min_overlap=0.5):
    """
    Merge transcription and diarization results
    
    Args:
        transcription_json: Path to transcription JSON
        diarization_json: Path to diarization JSON
        min_overlap: Minimum overlap ratio (default: 0.5 = 50%)
    
    Returns:
        Merged results dictionary
    """
    
    print(f"\n{'='*60}")
    print(f"Merging Transcription + Diarization")
    print(f"{'='*60}")
    print(f"Transcription: {transcription_json}")
    print(f"Diarization: {diarization_json}\n")
    
    # Load JSONs
    with open(transcription_json, 'r', encoding='utf-8') as f:
        transcription = json.load(f)
    
    with open(diarization_json, 'r', encoding='utf-8') as f:
        diarization = json.load(f)
    
    # Merge segments
    merged_segments = []
    skipped_count = 0
    unknown_count = 0
    
    for trans_seg in transcription['segments']:
        # Find speaker for this segment
        speaker = find_speaker_for_segment(
            trans_seg,
            diarization['segments'],
            min_overlap
        )
        
        # Skip empty segments
        if speaker is None:
            skipped_count += 1
            continue
        
        # Count unknown speakers
        if speaker == "Unknown":
            unknown_count += 1
        
        # Create merged segment
        merged_seg = {
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'duration': trans_seg['duration'],
            'speaker': speaker,
            'text': trans_seg['text'],
            'language': trans_seg['language'],
            'language_name': trans_seg.get('language_name', trans_seg['language'])
        }
        
        merged_segments.append(merged_seg)
    
    # Calculate statistics
    agent_segments = [s for s in merged_segments if s['speaker'] == 'Agent']
    user_segments = [s for s in merged_segments if s['speaker'] == 'User']
    
    agent_time = sum(s['duration'] for s in agent_segments)
    user_time = sum(s['duration'] for s in user_segments)
    
    # Count languages per speaker
    agent_languages = set(s['language'] for s in agent_segments)
    user_languages = set(s['language'] for s in user_segments)
    
    # Build results
    results = {
        'filename': transcription['filename'],
        'segments': merged_segments,
        'statistics': {
            'total_segments': len(merged_segments),
            'skipped_empty': skipped_count,
            'unknown_speaker': unknown_count,
            'agent_segments': len(agent_segments),
            'user_segments': len(user_segments),
            'agent_speaking_time': round(agent_time, 2),
            'user_speaking_time': round(user_time, 2),
            'agent_languages': sorted(list(agent_languages)),
            'user_languages': sorted(list(user_languages))
        },
        'languages_detected': transcription['languages_detected'],
        'is_multilingual': transcription['is_multilingual'],
        'merge_settings': {
            'min_overlap_ratio': min_overlap,
            'method': 'timestamp-overlap-matching'
        }
    }
    
    # Print summary
    print(f"{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"  Total segments: {len(merged_segments)}")
    print(f"  Skipped (empty): {skipped_count}")
    print(f"  Unknown speaker: {unknown_count}")
    print(f"\n  Agent: {len(agent_segments)} segments, {agent_time:.1f}s")
    print(f"    Languages: {', '.join(agent_languages)}")
    print(f"\n  User: {len(user_segments)} segments, {user_time:.1f}s")
    print(f"    Languages: {', '.join(user_languages)}")
    print(f"\n  Multilingual: {transcription['is_multilingual']}")
    print(f"  Languages: {', '.join(transcription['languages_detected'])}")
    
    # Show sample segments
    print(f"\nSample segments:")
    for seg in merged_segments[:5]:
        print(f"  [{seg['start']:.1f}s] ({seg['speaker']}) [{seg['language']}] {seg['text'][:60]}...")
    
    print(f"{'='*60}\n")
    
    return results


def save_results(results, output_path):
    """Save merged results to JSON"""
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to: {output_path}\n")


# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge transcription and diarization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python merge_results.py -t audio1_transcription.json -d audio1_diarization.json
  
  # With custom output
  python merge_results.py -t audio1_transcription.json -d audio1_diarization.json -o final.json
  
  # Adjust overlap threshold (60% minimum)
  python merge_results.py -t audio1_transcription.json -d audio1_diarization.json --min-overlap 0.6
        """
    )
    
    parser.add_argument('-t', '--transcription', type=str, required=True,
                        help='Transcription JSON file')
    parser.add_argument('-d', '--diarization', type=str, required=True,
                        help='Diarization JSON file')
    parser.add_argument('-o', '--output', type=str,
                        help='Output JSON file (default: outputs/final/<filename>_complete.json)')
    parser.add_argument('--min-overlap', type=float, default=0.5,
                        help='Minimum overlap ratio (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    # Merge
    results = merge_transcription_diarization(
        args.transcription,
        args.diarization,
        args.min_overlap
    )
    
    # Save with organized structure
    if args.output:
        output_path = args.output
    else:
        # Default: outputs/final/<filename>_complete.json
        output_dir = Path('outputs/final')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(results['filename']).stem
        output_path = output_dir / f"{filename}_complete.json"
    
    save_results(results, output_path)

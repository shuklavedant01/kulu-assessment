#!/usr/bin/env python3
"""
Voice Agent Latency Analyzer
Measures response times, turn-taking patterns, and conversation metrics
"""

import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_conversation(json_path):
    """Load merged conversation data with speaker labels and timestamps"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def calculate_turn_latencies(segments):
    """
    Calculate latency between User speech end and Agent response start
    
    Args:
        segments: List of conversation segments with speaker labels
    
    Returns:
        List of latency measurements with context
    """
    
    latencies = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        # User → Agent transition
        if current['speaker'] == 'User' and next_seg['speaker'] == 'Agent':
            latency = next_seg['start'] - current['end']
            
            latencies.append({
                'turn': len(latencies) + 1,
                'user_end': round(current['end'], 3),
                'agent_start': round(next_seg['start'], 3),
                'latency': round(latency, 3),
                'user_text': current['text'][:50] + '...' if len(current['text']) > 50 else current['text'],
                'agent_text': next_seg['text'][:50] + '...' if len(next_seg['text']) > 50 else next_seg['text']
            })
    
    return latencies


def calculate_statistics(latencies):
    """Calculate summary statistics for latency measurements"""
    
    if not latencies:
        return {}
    
    latency_values = [l['latency'] for l in latencies]
    
    stats = {
        'total_turns': len(latencies),
        'average': round(np.mean(latency_values), 3),
        'median': round(np.median(latency_values), 3),
        'min': round(np.min(latency_values), 3),
        'max': round(np.max(latency_values), 3),
        'std_dev': round(np.std(latency_values), 3),
        'p25': round(np.percentile(latency_values, 25), 3),
        'p75': round(np.percentile(latency_values, 75), 3),
        'p95': round(np.percentile(latency_values, 95), 3),
        'p99': round(np.percentile(latency_values, 99), 3) if len(latencies) > 20 else None
    }
    
    return stats


def calculate_additional_metrics(segments):
    """Calculate additional conversation quality metrics"""
    
    agent_segments = [s for s in segments if s['speaker'] == 'Agent']
    user_segments = [s for s in segments if s['speaker'] == 'User']
    
    # Speaking durations
    agent_total_time = sum(s['duration'] for s in agent_segments)
    user_total_time = sum(s['duration'] for s in user_segments)
    
    # Word counts (rough estimate: avg 150 words/minute)
    def estimate_words(text, duration):
        # Use actual word count if available, otherwise estimate from duration
        word_count = len(text.split())
        return word_count
    
    agent_words = sum(estimate_words(s['text'], s['duration']) for s in agent_segments)
    user_words = sum(estimate_words(s['text'], s['duration']) for s in user_segments)
    
    # Speaking rates (words per minute)
    agent_wpm = round((agent_words / agent_total_time) * 60, 1) if agent_total_time > 0 else 0
    user_wpm = round((user_words / user_total_time) * 60, 1) if user_total_time > 0 else 0
    
    # Detect overlaps (segments where timestamps overlap)
    overlaps = []
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        if current['end'] > next_seg['start']:
            overlap_duration = current['end'] - next_seg['start']
            overlaps.append({
                'time': round(next_seg['start'], 3),
                'duration': round(overlap_duration, 3),
                'speakers': f"{current['speaker']} ⟷ {next_seg['speaker']}"
            })
    
    # Silence/gap analysis
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]['start'] - segments[i]['end']
        if gap > 0:
            gaps.append(gap)
    
    metrics = {
        'agent_metrics': {
            'total_segments': len(agent_segments),
            'total_time': round(agent_total_time, 2),
            'avg_segment_duration': round(agent_total_time / len(agent_segments), 2) if agent_segments else 0,
            'total_words': agent_words,
            'words_per_minute': agent_wpm
        },
        'user_metrics': {
            'total_segments': len(user_segments),
            'total_time': round(user_total_time, 2),
            'avg_segment_duration': round(user_total_time / len(user_segments), 2) if user_segments else 0,
            'total_words': user_words,
            'words_per_minute': user_wpm
        },
        'conversation_metrics': {
            'total_duration': round(segments[-1]['end'], 2) if segments else 0,
            'overlaps_detected': len(overlaps),
            'overlaps': overlaps[:5] if overlaps else [],  # Show first 5
            'avg_gap': round(np.mean(gaps), 3) if gaps else 0,
            'gap_distribution': {
                '0-0.5s': sum(1 for g in gaps if 0 <= g < 0.5),
                '0.5-1s': sum(1 for g in gaps if 0.5 <= g < 1),
                '1-2s': sum(1 for g in gaps if 1 <= g < 2),
                '2s+': sum(1 for g in gaps if g >= 2)
            }
        }
    }
    
    return metrics


def visualize_latency_timeline(latencies, output_path):
    """Create line chart showing latency over conversation timeline"""
    
    if not latencies:
        print("⚠️  No latencies to plot")
        return
    
    turns = [l['turn'] for l in latencies]
    latency_values = [l['latency'] for l in latencies]
    
    plt.figure(figsize=(12, 6))
    plt.plot(turns, latency_values, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=np.mean(latency_values), color='r', linestyle='--', 
                label=f'Average: {np.mean(latency_values):.2f}s')
    
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.title('Agent Response Latency Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved timeline: {output_path}")


def visualize_latency_distribution(latencies, output_path):
    """Create histogram showing latency distribution"""
    
    if not latencies:
        print("⚠️  No latencies to plot")
        return
    
    latency_values = [l['latency'] for l in latencies]
    
    plt.figure(figsize=(10, 6))
    plt.hist(latency_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(latency_values), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(latency_values):.2f}s')
    plt.axvline(x=np.median(latency_values), color='g', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(latency_values):.2f}s')
    
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Agent Response Latency Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved distribution: {output_path}")


def save_results_json(latencies, stats, metrics, output_path):
    """Save all results to JSON file"""
    
    results = {
        'latency_analysis': {
            'turns': latencies,
            'statistics': stats
        },
        'additional_metrics': metrics,
        'summary': {
            'total_turns_analyzed': len(latencies),
            'average_latency': stats.get('average', 0),
            'conversation_duration': metrics['conversation_metrics']['total_duration']
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved JSON: {output_path}")


def save_results_csv(latencies, output_path):
    """Save turn-by-turn latencies to CSV"""
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['turn', 'user_end', 'agent_start', 
                                                'latency', 'user_text', 'agent_text'])
        writer.writeheader()
        writer.writerows(latencies)
    


def print_summary(latencies, stats, metrics):
    """Print comprehensive summary to console"""
    
    print(f"\n{'='*70}")
    print(f"LATENCY ANALYSIS SUMMARY")
    print(f"{'='*70}\n")
    
    # Latency Stats
    print(f"[STATS] Response Latency Statistics:")
    print(f"  Total Turns: {stats['total_turns']}")
    print(f"  Average: {stats['average']}s")
    print(f"  Median: {stats['median']}s")
    print(f"  Min: {stats['min']}s")
    print(f"  Max: {stats['max']}s")
    print(f"  Std Dev: {stats['std_dev']}s")
    print(f"  P95: {stats['p95']}s")
    
    # Speaker Metrics
    print(f"\n[AGENT] Agent Metrics:")
    agent = metrics['agent_metrics']
    print(f"  Speaking Time: {agent['total_time']}s")
    print(f"  Segments: {agent['total_segments']}")
    print(f"  Words/Min: {agent['words_per_minute']}")
    
    print(f"\n[USER] User Metrics:")
    user = metrics['user_metrics']
    print(f"  Speaking Time: {user['total_time']}s")
    print(f"  Segments: {user['total_segments']}")
    print(f"  Words/Min: {user['words_per_minute']}")
    
    # Conversation Quality
    print(f"\n[QUALITY] Conversation Quality:")
    conv = metrics['conversation_metrics']
    print(f"  Total Duration: {conv['total_duration']}s")
    print(f"  Overlaps Detected: {conv['overlaps_detected']}")
    print(f"  Avg Gap: {conv['avg_gap']}s")
    
    # Sample Turns
    print(f"\n[SAMPLES] Sample Turn Latencies:")
    for turn in latencies[:5]:
        # Use encode/decode to safely handle non-ASCII characters
        user_text = turn['user_text'].encode('ascii', 'replace').decode('ascii')
        agent_text = turn['agent_text'].encode('ascii', 'replace').decode('ascii')
        print(f"  Turn {turn['turn']}: {turn['latency']}s")
        print(f"    User: \"{user_text}\"")
        print(f"    Agent: \"{agent_text}\"")
    
    print(f"\n{'='*70}\n")




def analyze_conversation(input_json, output_dir="outputs/analysis"):
    """
    Complete latency analysis workflow
    
    Args:
        input_json: Path to merged conversation JSON
        output_dir: Directory for output files
    """
    
    print(f"{'='*70}")
    print(f"Voice Agent Latency Analyzer")
    print(f"{'='*70}")
    print(f"Input: {input_json}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load conversation
    segments = load_conversation(input_json)
    
    # Calculate latencies
    print("Analyzing turn latencies...")
    latencies = calculate_turn_latencies(segments)
    
    if not latencies:
        print("⚠️  No User→Agent transitions found!")
        return
    
    # Calculate statistics
    stats = calculate_statistics(latencies)
    
    # Calculate additional metrics
    print("Calculating additional metrics...")
    metrics = calculate_additional_metrics(segments)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    basename = Path(input_json).stem
    visualize_latency_timeline(latencies, output_path / f"{basename}_latency_timeline.png")
    visualize_latency_distribution(latencies, output_path / f"{basename}_latency_distribution.png")
    
    # Save results
    print("\nSaving results...")
    save_results_json(latencies, stats, metrics, output_path / f"{basename}_analysis.json")
    save_results_csv(latencies, output_path / f"{basename}_latencies.csv")
    
    # Print summary
    print_summary(latencies, stats, metrics)
    
    print(f"[OK] Analysis complete! Results saved to: {output_dir}")
    
    return {
        'latencies': latencies,
        'statistics': stats,
        'metrics': metrics
    }


# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze voice agent conversation latency and metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze conversation from merged JSON
  python latency_analyzer.py -f outputs/final/audio1_complete.json
  
  # Custom output directory
  python latency_analyzer.py -f outputs/final/audio1_complete.json -o my_analysis
        """
    )
    
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Merged conversation JSON file')
    parser.add_argument('-o', '--output', type=str, default='outputs/analysis',
                        help='Output directory for analysis results (default: outputs/analysis)')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_conversation(args.file, args.output)

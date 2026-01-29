#!/usr/bin/env python3
"""
Fully Adaptive Audio Cutout Detector
All thresholds learnedfrom conversation-specific baselines
"""

import json
import argparse
from pathlib import Path
import re
import statistics


def load_transcript(json_path):
    """Load conversation transcript with timestamps"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def load_diarization(json_path):
    """Load diarization data with speaker segments"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def calculate_response_baseline(segments):
    """
    Calculate conversation-specific baselines for ALL adaptive thresholds
    
    Returns dict with statistical metrics for:
    - user_to_agent: Agent response times (Rule 3)
    - all_gaps: All conversation gaps (Rule 1)
    - same_speaker_gaps: Intra-speaker pauses (Rule 2)
    """
    
    # Collect different types of gaps
    user_to_agent_gaps = []
    agent_to_user_gaps = []
    all_gaps = []
    same_speaker_gaps = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        gap = next_seg['start'] - current['end']
        
        # Collect all gaps (filter outliers > 15s)
        if gap < 15.0:
            all_gaps.append(gap)
        
        # User asks, Agent responds
        if current['speaker'] == 'User' and next_seg['speaker'] == 'Agent':
            user_to_agent_gaps.append(gap)
        
        # Agent speaks, User responds
        elif current['speaker'] == 'Agent' and next_seg['speaker'] == 'User':
            agent_to_user_gaps.append(gap)
        
        # Same speaker continues (thinking pauses)
        elif current['speaker'] == next_seg['speaker'] and gap < 10.0:
            same_speaker_gaps.append(gap)
    
    # Need minimum data
    if len(user_to_agent_gaps) < 3:
        return None
    
    sorted_gaps = sorted(user_to_agent_gaps)
    
    baseline = {
        'user_to_agent': {
            'sample_count': len(user_to_agent_gaps),
            'average': statistics.mean(user_to_agent_gaps),
            'median': statistics.median(user_to_agent_gaps),
            'min': min(user_to_agent_gaps),
            'max': max(user_to_agent_gaps),
            'p90': sorted_gaps[int(len(sorted_gaps) * 0.90)] if len(sorted_gaps) > 5 else sorted_gaps[-1],
            'p95': sorted_gaps[int(len(sorted_gaps) * 0.95)] if len(sorted_gaps) > 10 else sorted_gaps[-1],
            'stdev': statistics.stdev(user_to_agent_gaps) if len(user_to_agent_gaps) > 1 else 0
        }
    }
    
    # All gaps baseline (Rule 1: Long Silence)
    if len(all_gaps) >= 5:
        sorted_all = sorted(all_gaps)
        baseline['all_gaps'] = {
            'sample_count': len(all_gaps),
            'median': statistics.median(all_gaps),
            'p90': sorted_all[int(len(sorted_all) * 0.90)],
            'p95': sorted_all[int(len(sorted_all) * 0.95)] if len(sorted_all) > 10 else sorted_all[-1]
        }
    
    # Same-speaker gaps baseline (Rule 2: Incomplete Speech)
    if len(same_speaker_gaps) >= 3:
        baseline['same_speaker_gaps'] = {
            'sample_count': len(same_speaker_gaps),
            'median': statistics.median(same_speaker_gaps),
            'p90': sorted(same_speaker_gaps)[int(len(same_speaker_gaps) * 0.90)] if len(same_speaker_gaps) > 5 else max(same_speaker_gaps)
        }
    
    # Agent to user for context
    if len(agent_to_user_gaps) >= 3:
        sorted_gaps_user = sorted(agent_to_user_gaps)
        baseline['agent_to_user'] = {
            'sample_count': len(agent_to_user_gaps),
            'median': statistics.median(agent_to_user_gaps),
            'p90': sorted_gaps_user[int(len(sorted_gaps_user) * 0.90)] if len(sorted_gaps_user) > 5 else sorted_gaps_user[-1]
        }
    
    return baseline


def calculate_gaps(segments):
    """Calculate all gaps between consecutive segments"""
    gaps = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        gap_duration = next_seg['start'] - current['end']
        
        gaps.append({
            'index': i,
            'gap_start': current['end'],
            'gap_end': next_seg['start'],
            'duration': round(gap_duration, 3),
            'before_speaker': current['speaker'],
            'after_speaker': next_seg['speaker'],
            'before_text': current.get('text', ''),
            'after_text': next_seg.get('text', '')
        })
    
    return gaps


def is_incomplete_text(text):
    """Check if text ends incompletely (mid-sentence)"""
    text = text.strip()
    if not text:
        return False
    
    incomplete_endings = [
        'and', 'but', 'so', 'because', 'then', 'when', 'if',
        'the', 'a', 'an', 'to', 'for', 'with', 'about'
    ]
    
    words = text.split()
    if words:
        last_word = words[-1].lower().rstrip('.,!?')
        if last_word in incomplete_endings:
            return True
    
    if text.endswith('...') or text.endswith('-'):
        return True
    
    return False


def is_question(text):
    """Check if text is a question"""
    text = text.strip()
    
    if text.endswith('?'):
        return True
    
    question_words = ['what', 'where', 'when', 'why', 'who', 'how', 'can', 'could', 
                      'would', 'should', 'is', 'are', 'do', 'does', 'halo']
    first_word = text.split()[0].lower() if text.split() else ''
    if first_word in question_words:
        return True
    
    return False


def detect_missing_response_adaptive(gap, baseline):
    """
    Rule 3 ENHANCED (merged with Rule 1): Detect missing/delayed agent responses
    
    Detects when User speaks → Agent should respond → but doesn't (User speaks again)
    This covers:
    - Questions with no response
    - Statements with no acknowledgment  
    - Any conversation turn where Agent should engage
    
    CUTOUT = User → [gap] → User (Agent didn't respond)
    NOT FLAGGED = User → [gap] → Agent (latency, not cutout)
    """
    
    # User spoke, expecting Agent to respond
    if gap['before_speaker'] == 'User':
        
        # ONLY flag if Agent DIDN'T respond (User speaks again)
        if gap['after_speaker'] == 'User':
            
            # Use adaptive threshold if available
            if baseline and 'user_to_agent' in baseline:
                agent_baseline = baseline['user_to_agent']
                
                threshold_high = max(
                    agent_baseline['p95'] * 1.5,
                    agent_baseline['median'] * 2.5,
                    3.0
                )
                
                threshold_medium = max(
                    agent_baseline['p90'] * 1.3,
                    agent_baseline['median'] * 1.8,
                    2.0
                )
                
                # Determine type based on context
                is_question = gap['before_text'].strip().endswith('?') or \
                             any(gap['before_text'].lower().startswith(qw) for qw in 
                                 ['what', 'where', 'when', 'why', 'who', 'how', 'can', 'could'])
                
                cutout_type = 'missing_agent_response' if is_question else 'agent_no_acknowledgment'
                
                if gap['duration'] > threshold_high:
                    return {
                        'type': cutout_type,
                        'confidence': 'high',
                        'reason': f"User spoke but Agent didn't respond for {gap['duration']:.2f}s (baseline median: {agent_baseline['median']:.2f}s, p95: {agent_baseline['p95']:.2f}s)",
                        'baseline_info': {
                            'expected_median': round(agent_baseline['median'], 2),
                            'expected_p95': round(agent_baseline['p95'], 2),
                            'deviation': round(gap['duration'] / agent_baseline['median'], 2)
                        }
                    }
                elif gap['duration'] > threshold_medium:
                    return {
                        'type': cutout_type,
                        'confidence': 'medium',
                        'reason': f"User spoke but Agent didn't respond for {gap['duration']:.2f}s (baseline median: {agent_baseline['median']:.2f}s)",
                        'baseline_info': {
                            'expected_median': round(agent_baseline['median'], 2),
                            'deviation': round(gap['duration'] / agent_baseline['median'], 2)
                        }
                    }
            
            else:
                # Fallback to fixed threshold (only if User speaks again)
                if gap['duration'] > 2.0:
                    is_question = gap['before_text'].strip().endswith('?')
                    cutout_type = 'missing_agent_response' if is_question else 'agent_no_acknowledgment'
                    
                    return {
                        'type': cutout_type,
                        'confidence': 'high' if gap['duration'] > 5 else 'medium',
                        'reason': f"User spoke but Agent didn't respond for {gap['duration']}s (using fixed threshold)"
                    }
        
        # If Agent responds (gap['after_speaker'] == 'Agent'), this is LATENCY, not a cutout
        # → Don't flag anything, let latency_analyzer.py handle it
    
    return None


def detect_incomplete_speech_cutout(gap, baseline):
    """Rule 2 ADAPTIVE: Detect cutout after incomplete speech"""
    
   # Check if text is incomplete
    if not is_incomplete_text(gap['before_text']):
        return None
    
    # Use adaptive threshold if available
    if baseline and 'same_speaker_gaps' in baseline:
        pause_baseline = baseline['same_speaker_gaps']
        # For incomplete speech, even normal thinking pauses are suspicious
        # Threshold = max(median * 1.5, p90 * 1.2, 0.8 minimum)
        threshold = max(
            pause_baseline['median'] * 1.5,
            pause_baseline['p90'] * 1.2,
            0.8
        )
        
        if gap['duration'] > threshold:
            return {
                'type': 'incomplete_speech',
                'confidence': 'high' if gap['duration'] > threshold * 2 else 'medium',
                'reason': f"Speech ends incompletely with {gap['duration']:.2f}s gap (baseline median: {pause_baseline['median']:.2f}s)",
                'baseline_info': {
                    'expected_median': round(pause_baseline['median'], 2),
                    'threshold_used': round(threshold, 2)
                }
            }
    else:
        # Fallback to fixed threshold
        if gap['duration'] > 1.0:
            return {
                'type': 'incomplete_speech',
               'confidence': 'high' if gap['duration'] > 3 else 'medium',
                'reason': f"Speech ends incompletely: '{gap['before_text'][-50:]}' (using fixed threshold)"
            }
    
    return None


def detect_missing_response_adaptive(gap, baseline):
    """
    Rule 3 ADAPTIVE: Detect MISSING agent responses (cutouts only, not latency)
    
    CUTOUT = User asks question → Agent doesn't respond → User speaks again
    LATENCY = User asks question → Agent responds (even if slow) → Measured by latency_analyzer.py
    """
    
    # User asked a question
    if gap['before_speaker'] == 'User' and is_question(gap['before_text']):
        
        # Use adaptive threshold if available
        if baseline and 'user_to_agent' in baseline:
            agent_baseline = baseline['user_to_agent']
            
            threshold_high = max(
                agent_baseline['p95'] * 1.5,
                agent_baseline['median'] * 2.5,
                3.0
            )
            
            threshold_medium = max(
                agent_baseline['p90'] * 1.3,
                agent_baseline['median'] * 1.8,
                2.0
            )
            
            # ONLY flag if Agent DIDN'T respond (User speaks again)
            if gap['after_speaker'] == 'User':
                if gap['duration'] > threshold_high:
                    return {
                        'type': 'missing_agent_response',
                        'confidence': 'high',
                        'reason': f"User question ignored for {gap['duration']:.2f}s (baseline median: {agent_baseline['median']:.2f}s, p95: {agent_baseline['p95']:.2f}s)",
                        'baseline_info': {
                            'expected_median': round(agent_baseline['median'], 2),
                            'expected_p95': round(agent_baseline['p95'], 2),
                            'deviation': round(gap['duration'] / agent_baseline['median'], 2)
                        }
                    }
                elif gap['duration'] > threshold_medium:
                    return {
                        'type': 'missing_agent_response',
                        'confidence': 'medium',
                        'reason': f"User question ignored for {gap['duration']:.2f}s (baseline median: {agent_baseline['median']:.2f}s)",
                        'baseline_info': {
                            'expected_median': round(agent_baseline['median'], 2),
                            'deviation': round(gap['duration'] / agent_baseline['median'], 2)
                        }
                    }
            
            # If Agent responds (gap['after_speaker'] == 'Agent'), this is LATENCY, not a cutout
            # → Don't flag anything, let latency_analyzer.py handle it
        
        else:
            # Fallback to fixed threshold (only if User speaks again)
            if gap['after_speaker'] == 'User' and gap['duration'] > 2.0:
                return {
                    'type': 'missing_agent_response',
                    'confidence': 'high' if gap['duration'] > 5 else 'medium',
                    'reason': f"User question ignored for {gap['duration']}s (using fixed threshold)"
                }
    
    return None


def detect_mid_speech_cutout(gap, segments):
    """Rule 4 FIXED: Detect cutout during same speaker's speech"""
    if gap['before_speaker'] == gap['after_speaker']:
        if gap['duration'] > 2.0:
            if is_incomplete_text(gap['before_text']):
                return {
                    'type': 'mid_speech_interruption',
                    'confidence': 'high',
                    'reason': f"{gap['before_speaker']} interrupted for {gap['duration']}s mid-speech"
                }
    
    return None


def detect_user_response_delay(gap):
    """
    Rules 6-7 COMBINED: Monitor user response time (only when Agent is waiting)
    
    IMPORTANT: Only applies when Agent spoke last → User should respond
    Does NOT apply when User spoke last (User can think as long as they want)
    """
    
    # Only flag if Agent spoke and is waiting for User response
    if gap['before_speaker'] == 'Agent':
        
        # User is probably thinking/typing (1-5 minutes)
        if 60.0 < gap['duration'] <= 300.0:
            return {
                'type': 'user_may_be_thinking',
                'confidence': 'low',
                'reason': f"User hasn't responded in {gap['duration']:.0f}s ({gap['duration']/60:.1f} min), may be thinking or typing"
            }
        
        # User is probably disconnected/abandoned (5+ minutes)
        elif gap['duration'] > 300.0:
            return {
                'type': 'user_not_responding',
                'confidence': 'high',
                'reason': f"User hasn't responded in {gap['duration']:.0f}s ({gap['duration']/60:.1f} min), session likely abandoned"
            }
    
    return None


def is_natural_pause(gap):
    """Check if gap is likely a natural pause"""
    if gap['duration'] < 0.5:
        return True
    
    if gap['before_speaker'] != gap['after_speaker'] and gap['duration'] < 2.0:
        if not is_question(gap['before_text']):
            return True
    
    return False


def detect_within_sentence_dropouts(whisper_segments, diarization_segments):
    """Rule 5 FIXED: Detect micro-cutouts within sentences"""
    
    dropout_events = []
    
    for whisper_seg in whisper_segments:
        overlapping_diar = []
        for diar_seg in diarization_segments:
            overlap_start = max(whisper_seg['start'], diar_seg['start'])
            overlap_end = min(whisper_seg['end'], diar_seg['end'])
            overlap = overlap_end - overlap_start
            
            if overlap > 0:
                overlapping_diar.append(diar_seg)
        
        if len(overlapping_diar) > 1:
            for i in range(len(overlapping_diar) - 1):
                gap_start = overlapping_diar[i]['end']
                gap_end = overlapping_diar[i + 1]['start']
                gap_duration = gap_end - gap_start
                
                if gap_duration > 0.5:
                    dropout_events.append({
                        'gap_start': gap_start,
                        'gap_end': gap_end,
                        'duration': round(gap_duration, 3),
                        'type': 'within_sentence_dropout',
                        'confidence': 'high' if gap_duration > 1.0 else 'medium',
                        'reason': f"Audio dropout of {gap_duration:.2f}s within continuous speech",
                        'context': {
                            'speaker': whisper_seg['speaker'],
                            'text': whisper_seg.get('text', '')[:200],
                            'whisper_segment': f"{whisper_seg['start']:.2f}-{whisper_seg['end']:.2f}s",
                            'diarization_fragments': len(overlapping_diar),
                            'gap_position': f"Between fragments {i+1} and {i+2}"
                        }
                    })
    
    return dropout_events


def analyze_adaptive_cutouts(whisper_segments, diarization_segments=None):
    """Analyze conversation using fully adaptive detection"""
    
    # Step 1: Calculate baseline
    print("Calculating conversation baseline...")
    baseline = calculate_response_baseline(whisper_segments)
    
    if baseline:
        print(f"  Agent response baseline:")
        print(f"    Median: {baseline['user_to_agent']['median']:.2f}s")
        print(f"    P90: {baseline['user_to_agent']['p90']:.2f}s")
        print(f"    P95: {baseline['user_to_agent']['p95']:.2f}s")
        if 'all_gaps' in baseline:
            print(f"  All gaps baseline:")
            print(f"    Median: {baseline['all_gaps']['median']:.2f}s")
            print(f"    P90: {baseline['all_gaps']['p90']:.2f}s")
        if 'same_speaker_gaps' in baseline:
            print(f"  Same-speaker gaps baseline:")
            print(f"    Median: {baseline['same_speaker_gaps']['median']:.2f}s")
    else:
        print("  Insufficient data for baseline - using fixed thresholds")
    
    # Step 2: Sentence-level cutouts
    gaps = calculate_gaps(whisper_segments)
    sentence_level_cutouts = []
    
    for gap in gaps:
        if is_natural_pause(gap):
            continue
        
        detections = []
        
        # Rule 2: Incomplete speech
        result = detect_incomplete_speech_cutout(gap, baseline)
        if result:
            detections.append(result)
        
        # Rule 3 (merged with Rule 1): Missing agent response
        result = detect_missing_response_adaptive(gap, baseline)
        if result:
            detections.append(result)
        
        # Rule 4: Mid-speech cutout
        result = detect_mid_speech_cutout(gap, whisper_segments)
        if result:
            detections.append(result)
        
        # Rules 6-7 Combined: User response monitoring
        result = detect_user_response_delay(gap)
        if result:
            detections.append(result)
        
        if detections:
            best_detection = max(detections, key=lambda x: 
                {'high': 3, 'medium': 2, 'low': 1}.get(x['confidence'], 0))
            
            sentence_level_cutouts.append({
                'gap_start': gap['gap_start'],
                'gap_end': gap['gap_end'],
                'duration': gap['duration'],
                'type': best_detection['type'],
                'confidence': best_detection['confidence'],
                'reason': best_detection['reason'],
                'baseline_info': best_detection.get('baseline_info'),
                'context': {
                    'before_speaker': gap['before_speaker'],
                    'before_text': gap['before_text'][-100:] if len(gap['before_text']) > 100 else gap['before_text'],
                    'after_speaker': gap['after_speaker'],
                    'after_text': gap['after_text'][:100] if len(gap['after_text']) > 100 else gap['after_text']
                }
            })
    
    # Step 3: Within-sentence dropouts
    within_sentence_dropouts = []
    if diarization_segments:
        within_sentence_dropouts = detect_within_sentence_dropouts(whisper_segments, diarization_segments)
    
    # Combine and sort
    all_cutouts = sentence_level_cutouts + within_sentence_dropouts
    all_cutouts.sort(key=lambda x: x['gap_start'])
    
    return {
        'all_cutouts': all_cutouts,
        'sentence_level': sentence_level_cutouts,
        'within_sentence': within_sentence_dropouts,
        'baseline': baseline
    }


def calculate_statistics(cutout_events):
    """Calculate summary statistics"""
    if not cutout_events:
        return {
            'total_cutouts': 0,
            'by_type': {},
            'by_confidence': {},
            'total_lost_time': 0
        }
    
    by_type = {}
    for event in cutout_events:
        event_type = event['type']
        by_type[event_type] = by_type.get(event_type, 0) + 1
    
    by_confidence = {}
    for event in cutout_events:
        conf = event['confidence']
        by_confidence[conf] = by_confidence.get(conf, 0) + 1
    
    total_lost = sum(e['duration'] for e in cutout_events)
    
    return {
        'total_cutouts': len(cutout_events),
        'by_type': by_type,
        'by_confidence': by_confidence,
        'total_lost_time': round(total_lost, 2),
        'avg_cutout_duration': round(total_lost / len(cutout_events), 2) if cutout_events else 0
    }


def save_results(cutout_data, stats, output_path):
    """Save results to JSON"""
    results = {
        'summary': stats,
        'baseline_metrics': cutout_data.get('baseline'),
        'sentence_level_cutouts': len(cutout_data['sentence_level']),
        'within_sentence_dropouts': len(cutout_data['within_sentence']),
        'all_cutout_events': cutout_data['all_cutouts']
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved results: {output_path}")


def analyze_conversation_adaptive(transcript_json, diarization_json=None, output_dir="outputs/cutouts/adaptive"):
    """Complete adaptive cutout detection workflow"""
    
    print(f"{'='*70}")
    print(f"Fully Adaptive Audio Cutout Detector")
    print(f"{'='*70}")
    print(f"Transcript: {transcript_json}")
    if diarization_json:
        print(f"Diarization: {diarization_json}")
    print()
    
    # Load files
    whisper_segments = load_transcript(transcript_json)
    print(f"Loaded {len(whisper_segments)} transcription segments")
    
    diarization_segments = None
    if diarization_json:
        diarization_segments = load_diarization(diarization_json)
        print(f"Loaded {len(diarization_segments)} diarization segments")
    
    # Analyze
    print("\nAnalyzing for audio cutouts...")
    cutout_data = analyze_adaptive_cutouts(whisper_segments, diarization_segments)
    
    # Statistics
    stats = calculate_statistics(cutout_data['all_cutouts'])
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    basename = Path(transcript_json).stem
    save_results(cutout_data, stats, output_path / f"{basename}_adaptive_cutouts.json")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total Cutouts: {stats['total_cutouts']}")
    print(f"  Sentence-level: {len(cutout_data['sentence_level'])}")
    if diarization_segments:
        print(f"  Within-sentence: {len(cutout_data['within_sentence'])}")
    print(f"\nTypes:")
    for ctype, count in stats['by_type'].items():
        print(f"  {ctype}: {count}")
    print(f"\n[OK] Analysis complete! Results saved to: {output_dir}")
    
    return cutout_data, stats


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fully adaptive audio cutout detection with conversation-specific baselines'
    )
    
    parser.add_argument('-t', '--transcript', type=str, required=True,
                        help='Merged conversation JSON file')
    parser.add_argument('-d', '--diarization', type=str, default=None,
                        help='Diarization JSON file (optional, for within-sentence detection)')
    parser.add_argument('-o', '--output', type=str, default='outputs/cutouts/adaptive',
                        help='Output directory')
    
    args = parser.parse_args()
    
    analyze_conversation_adaptive(args.transcript, args.diarization, args.output)

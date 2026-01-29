#!/usr/bin/env python3
"""
Quick script to add gap_end timestamps to cutout events
"""

import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/cutouts/enhanced/audio2_complete_enhanced_cutouts.json"
output_file = input_file.replace('.json', '_with_endpoints.json')

# Load existing data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Add gap_end to each cutout event
for event in data['all_cutout_events']:
    if 'timestamp' in event:
        # Old format - convert
        event['gap_start'] = event.pop('timestamp')
        event['gap_end'] = round(event['gap_start'] + event['duration'], 3)
    elif 'gap_start' not in event:
        # Shouldn't happen but handle it
        print(f"Warning: Event missing gap_start: {event}")

# Save updated data
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✓ Updated {len(data['all_cutout_events'])} events")
print(f"✓ Saved to: {output_file}")

# Show sample
print(f"\nSample event:")
sample = data['all_cutout_events'][0]
print(f"  Gap: {sample['gap_start']}s → {sample['gap_end']}s (duration: {sample['duration']}s)")
print(f"  Type: {sample['type']}")

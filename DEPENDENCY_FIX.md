# Dependency Fix Guide

## Problem
`RuntimeError: operator torchvision::nms does not exist`

This occurs due to version incompatibility between torch, torchvision, and pyannote.audio.

## Solution

Run these commands in your terminal:

```bash
# Step 1: Uninstall conflicting packages
pip uninstall -y torch torchvision torchaudio pyannote.audio lightning pytorch-lightning torchmetrics

# Step 2: Install compatible versions
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Step 3: Install pyannote.audio
pip install pyannote.audio==3.1.1

# Step 4: Test
python -c "from pyannote.audio import Pipeline; print('Success!')"
```

## Quick Fix (Using Script)

```bash
chmod +x fix_dependencies.sh
./fix_dependencies.sh
```

## Verified Compatible Versions

- **torch**: 2.1.0
- **torchvision**: 0.16.0  
- **torchaudio**: 2.1.0
- **pyannote.audio**: 3.1.1

## After Fix

Run your diarization:
```bash
python diarization.py
```

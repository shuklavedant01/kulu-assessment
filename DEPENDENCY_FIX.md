# Dependency Fix Guide

## Problem
`RuntimeError: operator torchvision::nms does not exist`

This occurs due to version incompatibility between torch, torchvision, and pyannote.audio.

## Solution (Python 3.12 Compatible)

Run these commands in your terminal:

```bash
# Step 1: Uninstall conflicting packages
pip uninstall -y torch torchvision torchaudio pyannote.audio lightning pytorch-lightning torchmetrics

# Step 2: Install compatible versions (for Python 3.12)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Step 3: Install pyannote.audio
pip install pyannote.audio==3.1.1

# Step 4: Test
python -c "from pyannote.audio import Pipeline; print('✅ Success!')"
```

## Quick Fix (Using Script)

```bash
chmod +x fix_dependencies.sh
./fix_dependencies.sh
```

## Verified Compatible Versions (Python 3.12)

- **torch**: 2.2.0
- **torchvision**: 0.17.0  
- **torchaudio**: 2.2.0
- **pyannote.audio**: 3.1.1

> **Note**: Python 3.12 doesn't have access to torch 2.1.0. We use 2.2.0 instead which is compatible.

## After Fix

Run your diarization:
```bash
python diarization.py
```


# Dependency Fix Guide

## Problem
`RuntimeError: operator torchvision::nms does not exist`  
`AttributeError: np.NaN was removed in the NumPy 2.0 release`

This occurs due to version incompatibility between torch, torchvision, numpy, and pyannote.audio.

## Solution (Python 3.12 Compatible)

Run these commands in your terminal:

```bash
# Step 1: Uninstall conflicting packages
pip uninstall -y torch torchvision torchaudio pyannote.audio lightning pytorch-lightning torchmetrics numpy

<<<<<<< HEAD
# Step 2: Install compatible versions (for Python 3.12)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
=======
# Step 2: Install NumPy < 2.0 (pyannote.audio not compatible with NumPy 2.x)
pip install "numpy>=1.21.0,<2.0"
>>>>>>> 0788339 (Add diarization update)

# Step 3: Install compatible versions (for Python 3.12)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Step 4: Install pyannote.audio
pip install pyannote.audio==3.1.1

<<<<<<< HEAD
# Step 4: Test
python -c "from pyannote.audio import Pipeline; print('✅ Success!')"
=======
# Step 5: Test
python -c "from pyannote.audio import Pipeline; print('Success!')"
>>>>>>> 0788339 (Add diarization update)
```

## Quick Fix (Using Script)

```bash
chmod +x fix_dependencies.sh
./fix_dependencies.sh
```

## Verified Compatible Versions (Python 3.12)

<<<<<<< HEAD
=======
- **numpy**: 1.26.4 (or any >= 1.21.0, < 2.0)
>>>>>>> 0788339 (Add diarization update)
- **torch**: 2.2.0
- **torchvision**: 0.17.0  
- **torchaudio**: 2.2.0
- **pyannote.audio**: 3.1.1

<<<<<<< HEAD
> **Note**: Python 3.12 doesn't have access to torch 2.1.0. We use 2.2.0 instead which is compatible.
=======
> **Important Notes**: 
> - Python 3.12 doesn't have access to torch 2.1.0. We use 2.2.0 instead.
> - NumPy 2.x is NOT compatible with pyannote.audio 3.1.1 (uses deprecated `np.NaN`)
>>>>>>> 0788339 (Add diarization update)

## After Fix

Run your diarization:
```bash
python diarization.py
```


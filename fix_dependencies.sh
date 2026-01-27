#!/bin/bash
# Fix script for torchvision::nms error (Python 3.12 compatible)

echo "Uninstalling conflicting packages..."
pip uninstall -y torch torchvision torchaudio pyannote.audio lightning pytorch-lightning torchmetrics numpy

echo ""
echo "Installing compatible versions for Python 3.12..."

<<<<<<< HEAD
=======
# Install NumPy < 2.0 (pyannote.audio not compatible with NumPy 2.x)
pip install "numpy>=1.21.0,<2.0"

>>>>>>> 0788339 (Add diarization update)
# Install specific torch versions first (compatible with Python 3.12)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Install pyannote.audio
pip install pyannote.audio==3.1.1

echo ""
echo "Installation complete!"
echo ""
echo "Test with: python -c 'from pyannote.audio import Pipeline; print(\"Success!\")'"


#!/bin/bash
# Fix script for torchvision::nms error

echo "Uninstalling conflicting packages..."
pip uninstall -y torch torchvision torchaudio pyannote.audio lightning pytorch-lightning torchmetrics

echo ""
echo "Installing compatible versions..."

# Install specific torch versions first
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install pyannote.audio
pip install pyannote.audio==3.1.1

echo ""
echo "Installation complete!"
echo ""
echo "Test with: python -c 'from pyannote.audio import Pipeline; print(\"Success!\")'"

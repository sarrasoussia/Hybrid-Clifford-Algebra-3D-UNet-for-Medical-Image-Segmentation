# Geometry-Aware Medical Image Classification

Research experiment comparing baseline CNN with geometry-aware representations using MedMNIST.

## Overview

This project implements:
- Geometry-aware feature extraction using Sobel gradients (multivector representation)
- Baseline and geometry-aware CNN classifiers
- Grad-CAM visualization for interpretability
- Spatial coherence metrics (IoU, Dice)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Project Structure

- `dataset.py` - MedMNIST dataset loading
- `geometric_features.py` - Geometric feature extraction module
- `models.py` - CNN architectures (baseline and geometry-aware)
- `gradcam.py` - Grad-CAM implementation
- `metrics.py` - Spatial coherence metrics (IoU, Dice)
- `train.py` - Training and evaluation utilities
- `main.py` - Main experiment script

## Experiment Output

- Training/validation accuracy comparison
- Grad-CAM visualizations
- IoU and Dice coefficient measurements
- Saved model checkpoints
# Hybrid-Clifford-Algebra-3D-UNet-for-Medical-Image-Segmentation

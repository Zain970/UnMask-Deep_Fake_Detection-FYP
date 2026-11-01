# AI Model for DeepFake Detection

This directory contains the AI model implementation for the UnMask deepfake detection system.

## Directory Structure

```
ai_model/
├── models/              # Model architecture definitions
├── preprocessing/       # Image preprocessing utilities
└── requirements.txt     # Python dependencies
```

## Model Architecture

The model is implemented using PyTorch and consists of:

- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Dropout for regularization

## Preprocessing

The preprocessing module provides utilities for:

- Image loading and conversion
- Face detection
- Image normalization
- Batch processing

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Basic usage:

```python
from models.model_architecture import get_model
from preprocessing.preprocessing import ImagePreprocessor

# Initialize model and preprocessor
model = get_model()
preprocessor = ImagePreprocessor()

# Preprocess image
image_tensor = preprocessor.preprocess_image("path/to/image.jpg")

# Get prediction
prediction = model(image_tensor)
```

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)
- See requirements.txt for full dependency list

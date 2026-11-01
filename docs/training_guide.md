# Model Training Guide

## Overview
This guide provides detailed instructions for training the UnMask model, including data preparation, training configuration, and optimization techniques.

## Prerequisites

### Hardware Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM
- CPU: 4+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD

### Software Requirements
- Python 3.8+
- PyTorch 1.9.0+
- CUDA 11.0+ (for GPU training)
- Required Python packages (see requirements.txt)

## Data Preparation

### Dataset Structure
```
dataset/
├── train/
│   ├── real/
│   │   └── images/
│   └── fake/
│       └── images/
├── val/
│   ├── real/
│   │   └── images/
│   └── fake/
│       └── images/
└── test/
    ├── real/
    │   └── images/
    └── fake/
        └── images/
```

### Data Preprocessing
```python
from ai_model.preprocessing.image_preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(
    image_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Process single image
processed_image = preprocessor.preprocess_image(image)

# Process batch
processed_batch = preprocessor.preprocess_batch(batch)
```

## Training Configuration

### Model Configuration
```python
from ai_model.models.config import ModelConfig

config = ModelConfig(
    image_size=224,
    sequence_length=16,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=0.0001,
    attention_ratio=0.5
)
```

### Training Parameters
```python
training_params = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,
    'lr_scheduler': 'ReduceLROnPlateau',
    'optimizer': 'AdamW'
}
```

## Training Process

### 1. Initialize Model
```python
from ai_model.models.base_model import BaseModel

model = BaseModel(config)
model.to(device)  # Move to GPU if available
```

### 2. Set Up Training
```python
from ai_model.training.trainer import ModelTrainer

trainer = ModelTrainer(
    model=model,
    config=config,
    device=device
)
```

### 3. Start Training
```python
trainer.train(
    train_data=train_loader,
    val_data=val_loader,
    epochs=100,
    batch_size=32
)
```

## Training Monitoring

### Metrics Tracking
- Loss: CrossEntropyLoss
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### Visualization
```python
from ai_model.validation.validator import ModelValidator

validator = ModelValidator(model)
metrics = validator.validate(val_loader)
validator.visualize_results(metrics)
```

## Optimization Techniques

### 1. Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5
)
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Model Checkpointing

### Save Checkpoint
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'metrics': metrics
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Load Checkpoint
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Troubleshooting

### Common Issues
1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Adjust learning rate
   - Use gradient clipping
   - Check data preprocessing

3. **Poor Performance**
   - Verify data quality
   - Check model architecture
   - Adjust hyperparameters

### Performance Optimization
1. **Data Loading**
   - Use DataLoader with multiple workers
   - Enable pin_memory for GPU
   - Use prefetch_factor

2. **Model Optimization**
   - Enable cudnn benchmarking
   - Use torch.compile() (PyTorch 2.0+)
   - Optimize model architecture

## Best Practices

### Data Management
- Use data augmentation
- Implement proper validation split
- Monitor class balance
- Regular data quality checks

### Training Process
- Use early stopping
- Implement model checkpointing
- Monitor training metrics
- Regular validation checks

### Model Evaluation
- Use proper evaluation metrics
- Cross-validation when possible
- Test on diverse datasets
- Regular performance monitoring

## Version Control
- Track model versions
- Document training parameters
- Save training logs
- Maintain experiment records 
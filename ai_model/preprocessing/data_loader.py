import os
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

class DeepFakeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        is_train: bool = True,
        label_file: Optional[str] = None
    ):
        """
        Initialize the dataset
        
        Args:
            root_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
            is_train: Whether this is training data
            label_file: Path to CSV file containing labels
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load labels if provided
        self.labels = {}
        if label_file and os.path.exists(label_file):
            df = pd.read_csv(label_file)
            self.labels = dict(zip(df['filename'], df['label']))
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label if available
        label = self.labels.get(img_name, -1) if self.is_train else -1
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'label': label,
            'filename': img_name
        }
        
        return sample

def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
    label_file: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        train_transform: Transformations for training data
        val_transform: Transformations for validation data
        label_file: Path to CSV file containing labels
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = DeepFakeDataset(
        train_dir,
        transform=train_transform,
        is_train=True,
        label_file=label_file
    )
    
    val_dataset = DeepFakeDataset(
        val_dir,
        transform=val_transform,
        is_train=False,
        label_file=label_file
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def load_test_data(
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    """
    Create test data loader
    
    Args:
        test_dir: Directory containing test images
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        transform: Transformations for test data
        
    Returns:
        Test data loader
    """
    test_dataset = DeepFakeDataset(
        test_dir,
        transform=transform,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader 
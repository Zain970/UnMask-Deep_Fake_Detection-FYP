import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional, Tuple

class DataAugmentor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_albumentations: bool = True
    ):
        """
        Initialize the data augmentor
        
        Args:
            image_size: Target image size (height, width)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            use_albumentations: Whether to use albumentations for augmentation
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_albumentations = use_albumentations
        
        # Create augmentation pipelines
        self._create_pipelines()
    
    def _create_pipelines(self):
        """Create training and validation augmentation pipelines"""
        if self.use_albumentations:
            # Training augmentations
            self.train_transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.RandomResizedCrop(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    scale=(0.8, 1.0),
                    p=0.3
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
            
            # Validation augmentations
            self.val_transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            # Training augmentations using torchvision
            self.train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            
            # Validation augmentations using torchvision
            self.val_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
    
    def get_train_transform(self):
        """Get training augmentation pipeline"""
        return self.train_transform
    
    def get_val_transform(self):
        """Get validation augmentation pipeline"""
        return self.val_transform
    
    def apply_augmentation(
        self,
        image: torch.Tensor,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        Apply augmentation to a single image
        
        Args:
            image: Input image tensor
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented image tensor
        """
        transform = self.train_transform if is_training else self.val_transform
        
        if self.use_albumentations:
            # Convert to numpy for albumentations
            image_np = image.permute(1, 2, 0).numpy()
            augmented = transform(image=image_np)
            return augmented['image']
        else:
            return transform(image)
    
    def apply_batch_augmentation(
        self,
        batch: torch.Tensor,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        Apply augmentation to a batch of images
        
        Args:
            batch: Batch of image tensors
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented batch of image tensors
        """
        augmented_batch = []
        
        for image in batch:
            augmented_image = self.apply_augmentation(image, is_training)
            augmented_batch.append(augmented_image)
        
        return torch.stack(augmented_batch)
    
    def create_augmentation_pipeline(
        self,
        augmentations: List[str],
        probabilities: Optional[List[float]] = None
    ) -> A.Compose:
        """
        Create a custom augmentation pipeline
        
        Args:
            augmentations: List of augmentation names
            probabilities: Optional list of probabilities for each augmentation
            
        Returns:
            Custom augmentation pipeline
        """
        if not self.use_albumentations:
            raise ValueError("Custom pipelines only supported with albumentations")
        
        augmentation_list = []
        
        # Default probabilities if not provided
        if probabilities is None:
            probabilities = [0.5] * len(augmentations)
        
        # Map augmentation names to functions
        augmentation_map = {
            'flip': A.HorizontalFlip,
            'brightness': A.RandomBrightnessContrast,
            'gamma': A.RandomGamma,
            'noise': A.GaussNoise,
            'blur': A.GaussianBlur,
            'rotate': A.Rotate,
            'crop': A.RandomResizedCrop,
            'hue': A.HueSaturationValue,
            'sharpen': A.Sharpen,
            'elastic': A.ElasticTransform
        }
        
        # Create augmentation list
        for aug_name, prob in zip(augmentations, probabilities):
            if aug_name in augmentation_map:
                if aug_name == 'rotate':
                    augmentation_list.append(
                        augmentation_map[aug_name](limit=15, p=prob)
                    )
                elif aug_name == 'crop':
                    augmentation_list.append(
                        augmentation_map[aug_name](
                            height=self.image_size[0],
                            width=self.image_size[1],
                            scale=(0.8, 1.0),
                            p=prob
                        )
                    )
                else:
                    augmentation_list.append(augmentation_map[aug_name](p=prob))
        
        # Add normalization and tensor conversion
        augmentation_list.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        return A.Compose(augmentation_list) 
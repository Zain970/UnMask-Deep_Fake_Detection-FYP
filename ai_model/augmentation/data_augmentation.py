import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, Union
import random
import cv2
from PIL import Image, ImageEnhance

class GeometricTransformations:
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shear_range: Tuple[float, float] = (-5, 5),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        flip_prob: float = 0.5
    ):
        """
        Initialize geometric transformations
        
        Args:
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            shear_range: Range of shear angles in degrees
            translation_range: Range of translation factors
            flip_prob: Probability of horizontal flip
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        self.flip_prob = flip_prob
    
    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply geometric transformations
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            Transformed image tensor
        """
        # Convert to PIL Image for transformations
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Random rotation
        angle = random.uniform(*self.rotation_range)
        image = image.rotate(angle, resample=Image.BILINEAR)
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Random shear
        shear = random.uniform(*self.shear_range)
        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.BILINEAR
        )
        
        # Random translation
        tx = random.uniform(*self.translation_range) * width
        ty = random.uniform(*self.translation_range) * height
        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, tx, 0, 1, ty),
            resample=Image.BILINEAR
        )
        
        # Random flip
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert back to tensor
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        return image

class ColorAugmentations:
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        gamma_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize color augmentations
        
        Args:
            brightness_range: Range of brightness factors
            contrast_range: Range of contrast factors
            saturation_range: Range of saturation factors
            hue_range: Range of hue shifts
            gamma_range: Range of gamma correction factors
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gamma_range = gamma_range
    
    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply color augmentations
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        # Convert to PIL Image for color transformations
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Random brightness
        brightness_factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Random saturation
        saturation_factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        # Convert to HSV for hue shift
        image = np.array(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Random hue shift
        hue_shift = random.uniform(*self.hue_range) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back to RGB
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Random gamma correction
        gamma = random.uniform(*self.gamma_range)
        image = np.power(image / 255.0, gamma) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        image = Image.fromarray(image)
        
        # Convert back to tensor
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        return image

class NoiseInjection:
    def __init__(
        self,
        gaussian_noise_std: float = 0.1,
        salt_pepper_prob: float = 0.05,
        poisson_noise_scale: float = 0.1,
        speckle_noise_std: float = 0.1
    ):
        """
        Initialize noise injection
        
        Args:
            gaussian_noise_std: Standard deviation of Gaussian noise
            salt_pepper_prob: Probability of salt and pepper noise
            poisson_noise_scale: Scale of Poisson noise
            speckle_noise_std: Standard deviation of speckle noise
        """
        self.gaussian_noise_std = gaussian_noise_std
        self.salt_pepper_prob = salt_pepper_prob
        self.poisson_noise_scale = poisson_noise_scale
        self.speckle_noise_std = speckle_noise_std
    
    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply noise injection
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            Noisy image tensor
        """
        # Convert to numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Gaussian noise
        gaussian_noise = np.random.normal(
            0,
            self.gaussian_noise_std,
            image.shape
        )
        image = image + gaussian_noise
        
        # Salt and pepper noise
        salt_pepper = np.random.random(image.shape)
        image[salt_pepper < self.salt_pepper_prob / 2] = 0
        image[salt_pepper > 1 - self.salt_pepper_prob / 2] = 1
        
        # Poisson noise
        poisson_noise = np.random.poisson(
            image * self.poisson_noise_scale
        ) / self.poisson_noise_scale
        image = image + poisson_noise
        
        # Speckle noise
        speckle_noise = np.random.normal(
            0,
            self.speckle_noise_std,
            image.shape
        )
        image = image * (1 + speckle_noise)
        
        # Clip values
        image = np.clip(image, 0, 1)
        
        # Convert back to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        return image

class DataAugmentation:
    def __init__(
        self,
        use_geometric: bool = True,
        use_color: bool = True,
        use_noise: bool = True,
        geometric_params: Optional[Dict] = None,
        color_params: Optional[Dict] = None,
        noise_params: Optional[Dict] = None
    ):
        """
        Initialize data augmentation
        
        Args:
            use_geometric: Whether to use geometric transformations
            use_color: Whether to use color augmentations
            use_noise: Whether to use noise injection
            geometric_params: Parameters for geometric transformations
            color_params: Parameters for color augmentations
            noise_params: Parameters for noise injection
        """
        self.use_geometric = use_geometric
        self.use_color = use_color
        self.use_noise = use_noise
        
        # Initialize transformations
        if use_geometric:
            self.geometric = GeometricTransformations(
                **(geometric_params or {})
            )
        
        if use_color:
            self.color = ColorAugmentations(
                **(color_params or {})
            )
        
        if use_noise:
            self.noise = NoiseInjection(
                **(noise_params or {})
            )
    
    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply data augmentation
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        # Apply geometric transformations
        if self.use_geometric:
            image = self.geometric(image)
        
        # Apply color augmentations
        if self.use_color:
            image = self.color(image)
        
        # Apply noise injection
        if self.use_noise:
            image = self.noise(image)
        
        return image 
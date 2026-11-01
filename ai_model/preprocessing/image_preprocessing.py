import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List

class ImagePreprocessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the image preprocessor
        
        Args:
            image_size: Target image size (height, width)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # Basic transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def preprocess_image(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> torch.Tensor:
        """
        Preprocess a single image
        
        Args:
            image: Input image as numpy array
            face_bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            Preprocessed image as tensor
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Crop to face if bbox provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            image = image.crop((x, y, x + w, y + h))
        
        # Apply transforms
        image_tensor = self.base_transform(image)
        
        return image_tensor
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> torch.Tensor:
        """
        Preprocess a batch of images
        
        Args:
            images: List of input images
            face_bboxes: Optional list of face bounding boxes
            
        Returns:
            Batch of preprocessed images as tensor
        """
        processed_images = []
        
        for i, image in enumerate(images):
            bbox = face_bboxes[i] if face_bboxes else None
            processed_image = self.preprocess_image(image, bbox)
            processed_images.append(processed_image)
        
        return torch.stack(processed_images)
    
    def detect_faces(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image
            min_confidence: Minimum confidence for face detection
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist()
    
    def align_face(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Align face in image
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Aligned face image
        """
        x, y, w, h = face_bbox
        
        # Extract face region
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, self.image_size)
        
        return face
    
    def normalize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - np.array(self.mean)) / np.array(self.std)
        
        return image
    
    def denormalize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Denormalize image pixel values
        
        Args:
            image: Normalized image
            
        Returns:
            Original image
        """
        # Denormalize
        image = image * np.array(self.std) + np.array(self.mean)
        
        # Convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image 
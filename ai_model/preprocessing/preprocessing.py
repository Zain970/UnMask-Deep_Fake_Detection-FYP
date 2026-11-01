import cv2
import numpy as np
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for model input
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess a batch of images
        """
        batch_tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            batch_tensors.append(tensor)
        
        return torch.cat(batch_tensors, dim=0)
    
    def detect_face(self, image):
        """
        Detect face in image using OpenCV's face detector
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces 
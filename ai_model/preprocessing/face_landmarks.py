import cv2
import numpy as np
import dlib
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

class FaceLandmarkDetector:
    def __init__(
        self,
        predictor_path: str = 'shape_predictor_68_face_landmarks.dat',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the face landmark detector
        
        Args:
            predictor_path: Path to dlib's shape predictor model
            device: Device to run preprocessing on ('cuda' or 'cpu')
        """
        self.predictor = dlib.shape_predictor(predictor_path)
        self.device = device
        
        # Define landmark regions
        self.landmark_regions = {
            'jaw': list(range(0, 17)),
            'left_eyebrow': list(range(17, 22)),
            'right_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }
        
        # Initialize preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def detect_landmarks(
        self,
        image: np.ndarray,
        face_box: Tuple[int, int, int, int]
    ) -> Dict:
        """
        Detect facial landmarks in an image
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary containing:
            - landmarks: Array of landmark coordinates
            - regions: Dictionary of landmarks by region
        """
        # Convert face box to dlib rectangle
        x, y, w, h = face_box
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Detect landmarks
        landmarks = self.predictor(image, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Group landmarks by region
        regions = {}
        for region, indices in self.landmark_regions.items():
            regions[region] = landmarks[indices]
        
        return {
            'landmarks': landmarks,
            'regions': regions
        }
    
    def visualize_landmarks(
        self,
        image: np.ndarray,
        landmarks: Dict,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize facial landmarks on image
        
        Args:
            image: Input image
            landmarks: Landmark dictionary from detect_landmarks
            show: Whether to display the visualization
            save_path: Optional path to save visualization
            
        Returns:
            Image with landmarks visualized
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Draw landmarks
        for region, points in landmarks['regions'].items():
            # Convert points to integer coordinates
            points = points.astype(int)
            
            # Draw points
            for point in points:
                cv2.circle(vis_image, tuple(point), 2, (0, 255, 0), -1)
            
            # Draw lines between points
            if region in ['left_eyebrow', 'right_eyebrow', 'nose', 'left_eye', 'right_eye', 'mouth']:
                cv2.polylines(vis_image, [points], False, (0, 255, 0), 1)
        
        # Display visualization
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        # Save visualization
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def preprocess_landmarks(
        self,
        landmarks: Dict,
        image_size: Tuple[int, int] = (224, 224)
    ) -> Dict:
        """
        Preprocess landmarks for model input
        
        Args:
            landmarks: Landmark dictionary from detect_landmarks
            image_size: Target image size
            
        Returns:
            Preprocessed landmark dictionary
        """
        # Normalize coordinates to [0, 1]
        normalized_landmarks = {}
        for region, points in landmarks['regions'].items():
            # Normalize x coordinates
            points[:, 0] = points[:, 0] / image_size[0]
            # Normalize y coordinates
            points[:, 1] = points[:, 1] / image_size[1]
            normalized_landmarks[region] = points
        
        # Convert to tensors
        tensor_landmarks = {}
        for region, points in normalized_landmarks.items():
            tensor_landmarks[region] = torch.tensor(points, device=self.device)
        
        return {
            'landmarks': torch.tensor(landmarks['landmarks'], device=self.device),
            'regions': tensor_landmarks
        }
    
    def extract_landmark_features(
        self,
        landmarks: Dict
    ) -> Dict:
        """
        Extract features from landmarks
        
        Args:
            landmarks: Landmark dictionary from detect_landmarks
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Calculate eye aspect ratio
        left_eye = landmarks['regions']['left_eye']
        right_eye = landmarks['regions']['right_eye']
        features['eye_aspect_ratio'] = self._calculate_eye_aspect_ratio(left_eye, right_eye)
        
        # Calculate mouth aspect ratio
        mouth = landmarks['regions']['mouth']
        features['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(mouth)
        
        # Calculate nose width
        nose = landmarks['regions']['nose']
        features['nose_width'] = self._calculate_nose_width(nose)
        
        # Calculate face symmetry
        features['face_symmetry'] = self._calculate_face_symmetry(landmarks)
        
        return features
    
    def _calculate_eye_aspect_ratio(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> float:
        """Calculate eye aspect ratio"""
        # Calculate vertical distances
        left_vert = np.linalg.norm(left_eye[1] - left_eye[5])
        right_vert = np.linalg.norm(right_eye[1] - right_eye[5])
        
        # Calculate horizontal distances
        left_horz = np.linalg.norm(left_eye[0] - left_eye[3])
        right_horz = np.linalg.norm(right_eye[0] - right_eye[3])
        
        # Calculate aspect ratios
        left_ratio = left_vert / left_horz
        right_ratio = right_vert / right_horz
        
        return (left_ratio + right_ratio) / 2
    
    def _calculate_mouth_aspect_ratio(
        self,
        mouth: np.ndarray
    ) -> float:
        """Calculate mouth aspect ratio"""
        # Calculate vertical distance
        vert = np.linalg.norm(mouth[2] - mouth[6])
        
        # Calculate horizontal distance
        horz = np.linalg.norm(mouth[0] - mouth[4])
        
        return vert / horz
    
    def _calculate_nose_width(
        self,
        nose: np.ndarray
    ) -> float:
        """Calculate nose width"""
        return np.linalg.norm(nose[0] - nose[4])
    
    def _calculate_face_symmetry(
        self,
        landmarks: Dict
    ) -> float:
        """Calculate face symmetry score"""
        # Get left and right landmarks
        left_landmarks = np.concatenate([
            landmarks['regions']['left_eyebrow'],
            landmarks['regions']['left_eye']
        ])
        
        right_landmarks = np.concatenate([
            landmarks['regions']['right_eyebrow'],
            landmarks['regions']['right_eye']
        ])
        
        # Flip right landmarks
        right_landmarks[:, 0] = -right_landmarks[:, 0]
        
        # Calculate symmetry score
        symmetry_score = np.mean(np.abs(left_landmarks - right_landmarks))
        
        return 1.0 / (1.0 + symmetry_score)
    
    def batch_detect_landmarks(
        self,
        images: List[np.ndarray],
        face_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        Detect landmarks in a batch of images
        
        Args:
            images: List of input images
            face_boxes: List of face bounding boxes
            
        Returns:
            List of landmark dictionaries
        """
        results = []
        
        for image, box in zip(images, face_boxes):
            result = self.detect_landmarks(image, box)
            results.append(result)
        
        return results
    
    def batch_preprocess_landmarks(
        self,
        landmarks_list: List[Dict],
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[Dict]:
        """
        Preprocess a batch of landmarks
        
        Args:
            landmarks_list: List of landmark dictionaries
            image_size: Target image size
            
        Returns:
            List of preprocessed landmark dictionaries
        """
        results = []
        
        for landmarks in landmarks_list:
            result = self.preprocess_landmarks(landmarks, image_size)
            results.append(result)
        
        return results 
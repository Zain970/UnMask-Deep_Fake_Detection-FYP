import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import dlib

class FaceDetector:
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_face_size: int = 20,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        keep_all: bool = True
    ):
        """
        Initialize the face detector
        
        Args:
            device: Device to run MTCNN on ('cuda' or 'cpu')
            min_face_size: Minimum face size to detect
            thresholds: MTCNN thresholds for face detection
            factor: Scale factor for image pyramid
            post_process: Whether to post-process detected faces
            select_largest: Whether to select largest face
            keep_all: Whether to keep all detected faces
        """
        self.device = device
        self.mtcnn = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            select_largest=select_largest,
            keep_all=keep_all,
            device=device
        )
        
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    def detect_faces(
        self,
        image: np.ndarray,
        return_landmarks: bool = False
    ) -> Dict:
        """
        Detect faces in an image using MTCNN
        
        Args:
            image: Input image as numpy array
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            Dictionary containing:
            - boxes: Face bounding boxes
            - probs: Detection probabilities
            - landmarks: Facial landmarks (if return_landmarks=True)
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        result = {
            'boxes': boxes,
            'probs': probs
        }
        
        if return_landmarks:
            result['landmarks'] = landmarks
        
        return result
    
    def align_face(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        output_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Align face using facial landmarks
        
        Args:
            image: Input image
            face_box: Face bounding box
            landmarks: Facial landmarks
            output_size: Output image size
            
        Returns:
            Aligned face image
        """
        if landmarks is None:
            # Get landmarks if not provided
            landmarks = self._get_landmarks(image, face_box)
        
        # Get eye landmarks
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Crop face
        x, y, w, h = face_box.astype(int)
        face = rotated[y:y+h, x:x+w]
        
        # Resize to output size
        face = cv2.resize(face, output_size)
        
        return face
    
    def extract_face(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        margin: int = 20
    ) -> np.ndarray:
        """
        Extract face from image with margin
        
        Args:
            image: Input image
            face_box: Face bounding box
            margin: Margin around face
            
        Returns:
            Extracted face image
        """
        x, y, w, h = face_box.astype(int)
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        return face
    
    def _get_landmarks(
        self,
        image: np.ndarray,
        face_box: np.ndarray
    ) -> np.ndarray:
        """
        Get facial landmarks using dlib
        
        Args:
            image: Input image
            face_box: Face bounding box
            
        Returns:
            Facial landmarks
        """
        # Convert face box to dlib rectangle
        x, y, w, h = face_box.astype(int)
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get landmarks
        landmarks = self.predictor(image, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        return landmarks
    
    def batch_detect_faces(
        self,
        images: List[np.ndarray],
        return_landmarks: bool = False
    ) -> List[Dict]:
        """
        Detect faces in a batch of images
        
        Args:
            images: List of input images
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of detection results
        """
        results = []
        
        for image in images:
            result = self.detect_faces(image, return_landmarks)
            results.append(result)
        
        return results
    
    def batch_align_faces(
        self,
        images: List[np.ndarray],
        face_boxes: List[np.ndarray],
        landmarks: Optional[List[np.ndarray]] = None,
        output_size: Tuple[int, int] = (112, 112)
    ) -> List[np.ndarray]:
        """
        Align faces in a batch of images
        
        Args:
            images: List of input images
            face_boxes: List of face bounding boxes
            landmarks: List of facial landmarks
            output_size: Output image size
            
        Returns:
            List of aligned face images
        """
        aligned_faces = []
        
        for i, (image, box) in enumerate(zip(images, face_boxes)):
            land = landmarks[i] if landmarks else None
            aligned = self.align_face(image, box, land, output_size)
            aligned_faces.append(aligned)
        
        return aligned_faces
    
    def batch_extract_faces(
        self,
        images: List[np.ndarray],
        face_boxes: List[np.ndarray],
        margin: int = 20
    ) -> List[np.ndarray]:
        """
        Extract faces from a batch of images
        
        Args:
            images: List of input images
            face_boxes: List of face bounding boxes
            margin: Margin around faces
            
        Returns:
            List of extracted face images
        """
        extracted_faces = []
        
        for image, box in zip(images, face_boxes):
            face = self.extract_face(image, box, margin)
            extracted_faces.append(face)
        
        return extracted_faces 
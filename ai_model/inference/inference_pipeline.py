import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import torchvision.transforms as transforms

class InferencePipeline:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32,
        output_dir: str = 'inference_results'
    ):
        """
        Initialize inference pipeline
        
        Args:
            model: Model to use for inference
            device: Device to run inference on
            batch_size: Batch size for processing
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(
        self,
        image: Union[np.ndarray, Image.Image, str]
    ) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    
    def process_batch(
        self,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of images
        
        Args:
            batch: Batch of images
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            return predictions, probabilities
    
    def predict(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        class_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on list of images
        
        Args:
            images: List of input images
            class_names: List of class names
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process images in batches
        for i in tqdm(range(0, len(images), self.batch_size),
                     desc='Processing images'):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            batch = torch.cat(batch_tensors, dim=0)
            
            # Process batch
            predictions, probabilities = self.process_batch(batch)
            
            # Format results
            for j, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'image_index': i + j,
                    'prediction': pred.item(),
                    'confidence': prob[pred].item(),
                    'probabilities': prob.cpu().numpy().tolist()
                }
                
                if class_names is not None:
                    result['class_name'] = class_names[pred.item()]
                
                results.append(result)
        
        return results
    
    def predict_single(
        self,
        image: Union[np.ndarray, Image.Image, str],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run inference on single image
        
        Args:
            image: Input image
            class_names: List of class names
            
        Returns:
            Prediction result
        """
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # Process image
        predictions, probabilities = self.process_batch(tensor)
        
        # Format result
        result = {
            'prediction': predictions[0].item(),
            'confidence': probabilities[0][predictions[0]].item(),
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        if class_names is not None:
            result['class_name'] = class_names[predictions[0].item()]
        
        return result
    
    def save_results(
        self,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        filename: str = 'predictions.json'
    ):
        """
        Save inference results
        
        Args:
            results: Inference results
            filename: Output filename
        """
        # Add timestamp
        if isinstance(results, list):
            for result in results:
                result['timestamp'] = datetime.now().isoformat()
        else:
            results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f'Results saved to {self.output_dir / filename}')
    
    def visualize_results(
        self,
        image: Union[np.ndarray, Image.Image, str],
        result: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction results
        
        Args:
            image: Input image
            result: Prediction result
            save_path: Path to save visualization
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create visualization
        vis_image = image.copy()
        
        # Add prediction text
        text = f"Class: {result.get('class_name', result['prediction'])}"
        text += f" (Confidence: {result['confidence']:.2f})"
        
        cv2.putText(
            vis_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save or return visualization
        if save_path is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        else:
            return vis_image
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        frame_skip: int = 1,
        class_names: Optional[List[str]] = None
    ):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_skip: Process every nth frame
            class_names: List of class names
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        with tqdm(total=total_frames // frame_skip,
                 desc='Processing video') as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run inference
                    result = self.predict_single(frame_rgb, class_names)
                    
                    # Visualize results
                    vis_frame = self.visualize_results(frame_rgb, result)
                    
                    # Convert back to BGR and write
                    out.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                    
                    pbar.update(1)
                
                frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        self.logger.info(f'Video processing complete. Output saved to {output_path}')
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            class_names: List of class names
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = self.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(input_dir.glob(ext))
        
        # Process images
        results = []
        for img_path in tqdm(image_files, desc='Processing directory'):
            # Run inference
            result = self.predict_single(img_path, class_names)
            result['image_path'] = str(img_path)
            
            # Visualize results
            vis_path = output_dir / f"{img_path.stem}_result{img_path.suffix}"
            self.visualize_results(img_path, result, str(vis_path))
            
            results.append(result)
        
        # Save results
        self.save_results(results)
        
        self.logger.info(f'Directory processing complete. Results saved to {output_dir}') 
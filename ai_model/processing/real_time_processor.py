import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import queue
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cv2
from collections import deque
import psutil
import gc

class RealTimeProcessor:
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        buffer_size: int = 30,
        num_workers: int = 4,
        batch_size: int = 8,
        output_dir: str = 'processing_results'
    ):
        """
        Initialize real-time processor
        
        Args:
            model: Model for processing
            input_shape: Input tensor shape
            device: Device to run processing on
            buffer_size: Frame buffer size
            num_workers: Number of parallel workers
            batch_size: Processing batch size
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.input_shape = input_shape
        self.device = device
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.frame_buffer = deque(maxlen=buffer_size)
        self.result_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Set model to eval mode
        self.model.eval()
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize processing workers"""
        # Thread pool for frame preprocessing
        self.preprocess_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Process pool for model inference
        self.inference_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        
        # Start processing threads
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self.processing_thread.start()
    
    def add_frame(self, frame: np.ndarray):
        """
        Add frame to processing pipeline
        
        Args:
            frame: Input frame
        """
        # Preprocess frame
        future = self.preprocess_pool.submit(self._preprocess_frame, frame)
        self.processing_queue.put(future)
    
    def _preprocess_frame(
        self,
        frame: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess frame
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed tensor
        """
        # Convert to tensor
        tensor = torch.from_numpy(frame).float()
        
        # Normalize
        tensor = tensor / 255.0
        
        # Resize if needed
        if tensor.shape != self.input_shape:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=self.input_shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return tensor
    
    def _process_frames(self):
        """Process frames in background"""
        batch = []
        batch_futures = []
        
        while not self.stop_event.is_set():
            try:
                # Get next frame
                future = self.processing_queue.get(timeout=0.1)
                tensor = future.result()
                
                # Add to batch
                batch.append(tensor)
                batch_futures.append(future)
                
                # Process batch if full
                if len(batch) >= self.batch_size:
                    self._process_batch(batch, batch_futures)
                    batch = []
                    batch_futures = []
                
            except queue.Empty:
                # Process remaining frames
                if batch:
                    self._process_batch(batch, batch_futures)
                    batch = []
                    batch_futures = []
                
            except Exception as e:
                self.logger.error(f"Error processing frames: {e}")
    
    def _process_batch(
        self,
        batch: List[torch.Tensor],
        futures: List[Any]
    ):
        """
        Process batch of frames
        
        Args:
            batch: List of frame tensors
            futures: List of futures
        """
        # Stack batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # Post-process results
        results = self._postprocess_outputs(outputs)
        
        # Put results in queue
        for result in results:
            self.result_queue.put(result)
    
    def _postprocess_outputs(
        self,
        outputs: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Post-process model outputs
        
        Args:
            outputs: Model outputs
            
        Returns:
            List of processed results
        """
        results = []
        
        # Convert to numpy
        outputs_np = outputs.cpu().numpy()
        
        # Process each output
        for output in outputs_np:
            result = {
                'prediction': np.argmax(output),
                'confidence': float(np.max(output)),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        return results
    
    def get_results(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get processing results
        
        Args:
            timeout: Queue timeout
            
        Returns:
            Processing result or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def optimize_performance(self):
        """Optimize processing performance"""
        # Set model to eval mode
        self.model.eval()
        
        # Enable CUDA optimization
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Optimize memory usage
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Set process priority
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    
    def monitor_performance(self) -> Dict[str, float]:
        """
        Monitor processing performance
        
        Returns:
            Performance metrics
        """
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'queue_size': self.processing_queue.qsize(),
            'buffer_size': len(self.frame_buffer),
            'result_queue_size': self.result_queue.qsize()
        }
        
        if self.device == 'cuda':
            metrics['gpu_usage'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            metrics['gpu_memory'] = torch.cuda.memory_reserved() / torch.cuda.max_memory_reserved()
        
        return metrics
    
    def stop(self):
        """Stop processing"""
        self.stop_event.set()
        self.preprocess_pool.shutdown()
        self.inference_pool.shutdown()
        self.processing_thread.join()
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        filename: str = 'processing_results.json'
    ):
        """
        Save processing results
        
        Args:
            results: List of results
            filename: Output filename
        """
        # Save results
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f'Results saved to {self.output_dir / filename}')
    
    def load_results(
        self,
        filename: str = 'processing_results.json'
    ) -> List[Dict[str, Any]]:
        """
        Load processing results
        
        Args:
            filename: Input filename
            
        Returns:
            List of results
        """
        # Load results
        with open(self.output_dir / filename, 'r') as f:
            results = json.load(f)
        
        return results 
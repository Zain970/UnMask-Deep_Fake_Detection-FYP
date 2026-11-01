import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
import redis
import pickle
import base64
import os
from typing import Optional

class ModelServer:
    """
    A class for serving deep learning models with REST API endpoints,
    batch processing, and result caching.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        host: str = "0.0.0.0",
        port: int = 8000,
        cache_size: int = 1000,
        batch_size: int = 32,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        """
        Initialize the model server.
        
        Args:
            model: The model to serve
            device: Device to run the model on
            host: Host address to serve on
            port: Port to serve on
            cache_size: Size of the LRU cache
            batch_size: Size of batches for processing
            redis_host: Redis host address
            redis_port: Redis port
            redis_db: Redis database number
        """
        self.model = model
        self.device = device
        self.host = host
        self.port = port
        self.batch_size = batch_size
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Model Server")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )
        
        # Initialize thread pool for batch processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('model_server.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Define request models
        class PredictionRequest(BaseModel):
            data: List[Dict[str, Any]]
            batch_id: Optional[str] = None
        
        class BatchStatusRequest(BaseModel):
            batch_id: str
        
        # Register routes
        @self.app.post("/predict")
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            try:
                if len(request.data) > self.batch_size:
                    # Process as batch
                    batch_id = request.batch_id or str(uuid.uuid4())
                    background_tasks.add_task(
                        self._process_batch,
                        request.data,
                        batch_id
                    )
                    return {"batch_id": batch_id, "status": "processing"}
                else:
                    # Process immediately
                    results = await self._process_single(request.data)
                    return {"results": results}
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/batch_status/{batch_id}")
        async def batch_status(batch_id: str):
            try:
                status = self._get_batch_status(batch_id)
                return {"status": status}
            except Exception as e:
                self.logger.error(f"Status check error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/batch_results/{batch_id}")
        async def batch_results(batch_id: str):
            try:
                results = self._get_batch_results(batch_id)
                return {"results": results}
            except Exception as e:
                self.logger.error(f"Results retrieval error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_single(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a single request.
        
        Args:
            data: Input data
            
        Returns:
            List of prediction results
        """
        # Check cache
        cache_key = self._generate_cache_key(data)
        cached_results = self._get_from_cache(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Process data
        results = []
        for item in data:
            # Preprocess input
            input_tensor = self._preprocess_input(item)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = self._postprocess_output(output)
            
            results.append(prediction)
        
        # Cache results
        self._cache_results(cache_key, results)
        
        return results
    
    async def _process_batch(
        self,
        data: List[Dict[str, Any]],
        batch_id: str
    ) -> None:
        """
        Process a batch of requests.
        
        Args:
            data: Input data
            batch_id: Batch identifier
        """
        try:
            # Update status
            self._update_batch_status(batch_id, "processing")
            
            # Process in chunks
            results = []
            for i in range(0, len(data), self.batch_size):
                chunk = data[i:i + self.batch_size]
                chunk_results = await self._process_single(chunk)
                results.extend(chunk_results)
            
            # Save results
            self._save_batch_results(batch_id, results)
            
            # Update status
            self._update_batch_status(batch_id, "completed")
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            self._update_batch_status(batch_id, "failed")
    
    def _preprocess_input(
        self,
        data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Preprocess input data.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed tensor
        """
        # Implement preprocessing logic
        # This is a placeholder - implement according to your model's needs
        return torch.tensor(data["input"]).to(self.device)
    
    def _postprocess_output(
        self,
        output: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Postprocess model output.
        
        Args:
            output: Model output
            
        Returns:
            Postprocessed results
        """
        # Implement postprocessing logic
        # This is a placeholder - implement according to your model's needs
        return {
            "prediction": output.argmax().item(),
            "confidence": output.softmax(dim=0).max().item()
        }
    
    def _generate_cache_key(
        self,
        data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate cache key for data.
        
        Args:
            data: Input data
            
        Returns:
            Cache key
        """
        return base64.b64encode(
            pickle.dumps(data)
        ).decode('utf-8')
    
    def _get_from_cache(
        self,
        key: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get results from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached results or None
        """
        cached = self.redis_client.get(key)
        if cached is not None:
            return pickle.loads(cached)
        return None
    
    def _cache_results(
        self,
        key: str,
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Cache results.
        
        Args:
            key: Cache key
            results: Results to cache
        """
        self.redis_client.set(
            key,
            pickle.dumps(results),
            ex=3600  # Cache for 1 hour
        )
    
    def _update_batch_status(
        self,
        batch_id: str,
        status: str
    ) -> None:
        """
        Update batch processing status.
        
        Args:
            batch_id: Batch identifier
            status: New status
        """
        self.redis_client.set(
            f"batch_status:{batch_id}",
            status
        )
    
    def _get_batch_status(
        self,
        batch_id: str
    ) -> str:
        """
        Get batch processing status.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch status
        """
        status = self.redis_client.get(f"batch_status:{batch_id}")
        return status.decode('utf-8') if status else "unknown"
    
    def _save_batch_results(
        self,
        batch_id: str,
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Save batch results.
        
        Args:
            batch_id: Batch identifier
            results: Results to save
        """
        self.redis_client.set(
            f"batch_results:{batch_id}",
            pickle.dumps(results)
        )
    
    def _get_batch_results(
        self,
        batch_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get batch results.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch results
        """
        results = self.redis_client.get(f"batch_results:{batch_id}")
        if results is None:
            raise HTTPException(status_code=404, detail="Results not found")
        return pickle.loads(results)
    
    def start(self) -> None:
        """
        Start the model server.
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        ) 
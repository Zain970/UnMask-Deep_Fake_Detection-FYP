from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path
from fastapi import HTTPException
from .redis import get_redis
from .storage import storage_service

class ProcessingService:
    def __init__(self):
        self.redis = get_redis()
        self.status_key_prefix = "processing:status:"
        self.result_key_prefix = "processing:result:"
        self.error_key_prefix = "processing:error:"
    
    async def start_processing(self, file_path: str, user_id: int) -> str:
        """Start processing a file and return processing ID"""
        processing_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Initialize status
        status = {
            "status": "pending",
            "started_at": datetime.now().isoformat(),
            "file_path": file_path,
            "user_id": user_id,
            "progress": 0
        }
        
        # Store initial status
        await self.redis.set(
            f"{self.status_key_prefix}{processing_id}",
            json.dumps(status)
        )
        
        return processing_id
    
    async def update_status(
        self,
        processing_id: str,
        status: str,
        progress: int,
        error: Optional[str] = None
    ) -> None:
        """Update processing status"""
        current_status = await self.get_status(processing_id)
        if not current_status:
            raise HTTPException(status_code=404, detail="Processing not found")
        
        current_status["status"] = status
        current_status["progress"] = progress
        current_status["updated_at"] = datetime.now().isoformat()
        
        if error:
            current_status["error"] = error
            await self.redis.set(
                f"{self.error_key_prefix}{processing_id}",
                error
            )
        
        await self.redis.set(
            f"{self.status_key_prefix}{processing_id}",
            json.dumps(current_status)
        )
    
    async def save_result(
        self,
        processing_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Save processing result"""
        await self.redis.set(
            f"{self.result_key_prefix}{processing_id}",
            json.dumps(result)
        )
        
        # Update status to completed
        await self.update_status(processing_id, "completed", 100)
    
    async def get_status(self, processing_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status"""
        status_data = await self.redis.get(f"{self.status_key_prefix}{processing_id}")
        if status_data:
            return json.loads(status_data)
        return None
    
    async def get_result(self, processing_id: str) -> Optional[Dict[str, Any]]:
        """Get processing result"""
        result_data = await self.redis.get(f"{self.result_key_prefix}{processing_id}")
        if result_data:
            return json.loads(result_data)
        return None
    
    async def get_error(self, processing_id: str) -> Optional[str]:
        """Get processing error"""
        return await self.redis.get(f"{self.error_key_prefix}{processing_id}")
    
    async def cleanup(self, processing_id: str) -> None:
        """Clean up processing data"""
        await self.redis.delete(
            f"{self.status_key_prefix}{processing_id}",
            f"{self.result_key_prefix}{processing_id}",
            f"{self.error_key_prefix}{processing_id}"
        )

processing_service = ProcessingService() 
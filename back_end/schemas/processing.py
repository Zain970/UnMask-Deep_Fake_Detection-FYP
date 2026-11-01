from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ProcessingStatus(BaseModel):
    status: str
    started_at: datetime
    updated_at: Optional[datetime] = None
    file_path: str
    user_id: int
    progress: int
    error: Optional[str] = None

class ProcessingResult(BaseModel):
    processing_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

class ProcessingRequest(BaseModel):
    file_path: str
    options: Optional[Dict[str, Any]] = None

class ProcessingResponse(BaseModel):
    processing_id: str
    status: str
    message: str 
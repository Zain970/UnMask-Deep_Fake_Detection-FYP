from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...core.security import get_current_user
from ...core.processing import processing_service
from ...schemas.processing import (
    ProcessingRequest,
    ProcessingResponse,
    ProcessingStatus,
    ProcessingResult
)
from ...models.user import User

router = APIRouter()

@router.post("/process", response_model=ProcessingResponse)
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start processing a file"""
    try:
        # Start processing
        processing_id = await processing_service.start_processing(
            request.file_path,
            current_user.id
        )
        
        # Add background task for processing
        background_tasks.add_task(
            process_file,
            processing_id,
            request.file_path,
            request.options
        )
        
        return ProcessingResponse(
            processing_id=processing_id,
            status="pending",
            message="Processing started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{processing_id}", response_model=ProcessingStatus)
async def get_processing_status(
    processing_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get processing status"""
    status = await processing_service.get_status(processing_id)
    if not status:
        raise HTTPException(status_code=404, detail="Processing not found")
    
    if status["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return ProcessingStatus(**status)

@router.get("/result/{processing_id}", response_model=ProcessingResult)
async def get_processing_result(
    processing_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get processing result"""
    status = await processing_service.get_status(processing_id)
    if not status:
        raise HTTPException(status_code=404, detail="Processing not found")
    
    if status["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    result = await processing_service.get_result(processing_id)
    error = await processing_service.get_error(processing_id)
    
    return ProcessingResult(
        processing_id=processing_id,
        status=status["status"],
        result=result,
        error=error,
        started_at=status["started_at"],
        completed_at=status.get("updated_at") if status["status"] == "completed" else None
    )

@router.get("/history", response_model=List[ProcessingResult])
async def get_processing_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get processing history for current user"""
    # TODO: Implement history retrieval from database
    return []

async def process_file(
    processing_id: str,
    file_path: str,
    options: dict = None
):
    """Background task for processing file"""
    try:
        # Update status to processing
        await processing_service.update_status(
            processing_id,
            "processing",
            10
        )
        
        # TODO: Implement actual file processing
        # This is a placeholder for the actual processing logic
        await processing_service.update_status(
            processing_id,
            "processing",
            50
        )
        
        # Simulate processing completion
        result = {
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "confidence": 0.95,
            "is_deepfake": True
        }
        
        await processing_service.save_result(processing_id, result)
        
    except Exception as e:
        await processing_service.update_status(
            processing_id,
            "failed",
            0,
            str(e)
        ) 
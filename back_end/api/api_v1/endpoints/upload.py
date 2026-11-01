from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import List
import os
from datetime import datetime
from ...core.storage import storage_service
from ...schemas.upload import FileUploadResponse, FileDeleteResponse, FileInfo
from ...core.security import oauth2_scheme

router = APIRouter()

@router.post("/upload/image", response_model=FileUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    """Upload an image file"""
    try:
        file_path = await storage_service.save_upload_file(file, "image")
        file_size = os.path.getsize(file_path)
        
        return FileUploadResponse(
            file_path=file_path,
            file_type="image",
            file_name=file.filename,
            file_size=file_size,
            message="Image uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/video", response_model=FileUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    """Upload a video file"""
    try:
        file_path = await storage_service.save_upload_file(file, "video")
        file_size = os.path.getsize(file_path)
        
        return FileUploadResponse(
            file_path=file_path,
            file_type="video",
            file_name=file.filename,
            file_size=file_size,
            message="Video uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/file/{file_path:path}", response_model=FileDeleteResponse)
async def delete_file(
    file_path: str,
    token: str = Depends(oauth2_scheme)
):
    """Delete a file"""
    try:
        success = storage_service.delete_file(file_path)
        if success:
            return FileDeleteResponse(
                success=True,
                message="File deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files", response_model=List[FileInfo])
async def list_files(
    file_type: str = "image",
    token: str = Depends(oauth2_scheme)
):
    """List all uploaded files of a specific type"""
    try:
        if file_type == "image":
            target_dir = storage_service.image_dir
        elif file_type == "video":
            target_dir = storage_service.video_dir
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        files = []
        for file_path in target_dir.glob("*"):
            if file_path.is_file():
                files.append(FileInfo(
                    file_path=str(file_path),
                    file_type=file_type,
                    file_name=file_path.name,
                    file_size=file_path.stat().st_size,
                    created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                ))
        
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
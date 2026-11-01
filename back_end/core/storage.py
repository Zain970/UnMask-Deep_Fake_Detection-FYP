import os
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException
from ..config import settings

class StorageService:
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.image_dir = self.upload_dir / "images"
        self.video_dir = self.upload_dir / "videos"
        self.temp_dir = self.upload_dir / "temp"
        
        for directory in [self.image_dir, self.video_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
    
    async def save_upload_file(self, file: UploadFile, file_type: str = "image") -> str:
        """Save uploaded file and return its path"""
        try:
            # Validate file type
            if file_type == "image":
                if not file.content_type.startswith("image/"):
                    raise HTTPException(status_code=400, detail="File must be an image")
                target_dir = self.image_dir
            elif file_type == "video":
                if not file.content_type.startswith("video/"):
                    raise HTTPException(status_code=400, detail="File must be a video")
                target_dir = self.video_dir
            else:
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{os.urandom(8).hex()}{file_extension}"
            file_path = target_dir / unique_filename
            
            # Save file
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            return str(file_path)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file from storage"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def get_file_path(self, filename: str, file_type: str = "image") -> Optional[Path]:
        """Get the full path of a file"""
        if file_type == "image":
            target_dir = self.image_dir
        elif file_type == "video":
            target_dir = self.video_dir
        else:
            return None
        
        file_path = target_dir / filename
        return file_path if file_path.exists() else None

storage_service = StorageService() 
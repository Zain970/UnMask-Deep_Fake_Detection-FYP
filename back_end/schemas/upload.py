from pydantic import BaseModel
from typing import Optional

class FileUploadResponse(BaseModel):
    file_path: str
    file_type: str
    file_name: str
    file_size: int
    message: Optional[str] = None

class FileDeleteResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class FileInfo(BaseModel):
    file_path: str
    file_type: str
    file_name: str
    file_size: int
    created_at: str 
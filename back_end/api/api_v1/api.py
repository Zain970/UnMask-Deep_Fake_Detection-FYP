from fastapi import APIRouter
from .endpoints import auth, users, predictions, models, upload, processing

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(upload.router, prefix="/files", tags=["files"])
api_router.include_router(processing.router, prefix="/processing", tags=["processing"]) 
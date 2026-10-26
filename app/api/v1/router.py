from fastapi import APIRouter

from app.api.v1.endpoints import check_ins, faces, health, recognition, stream

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(faces.router)
api_router.include_router(recognition.router)
api_router.include_router(stream.router)
api_router.include_router(check_ins.router)

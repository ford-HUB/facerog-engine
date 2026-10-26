from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    execution_providers: list[str]
    enrolled_people: int
    stream_running: bool
    video_source_open: bool


@router.get("/health", response_model=HealthResponse)
def get_health(container: Annotated[ServiceContainer, Depends(get_container)]) -> HealthResponse:
    snapshot = container.tracking_service.get_snapshot()
    return HealthResponse(
        status="ok",
        execution_providers=list(container.execution_providers),
        enrolled_people=len(container.enrollment_service.list_faces()),
        stream_running=snapshot.is_running,
        video_source_open=snapshot.is_source_open,
    )

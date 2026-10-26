from datetime import datetime

from pydantic import BaseModel


class FaceResponse(BaseModel):
    person_id: str
    name: str
    photo_url: str
    image_count: int
    template_count: int
    embedding_model: str
    enrolled_at: datetime


class FaceListResponse(BaseModel):
    total: int
    faces: list[FaceResponse]


class EnrollmentSummaryResponse(BaseModel):
    person_count: int
    template_count: int
    skipped_images: list[str]

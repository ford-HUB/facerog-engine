from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import (
    get_enrollment_service,
    get_recognition_service,
    get_settings_dependency,
)
from app.api.uploads import read_image_upload
from app.api.v1.presenters import present_recognition
from app.core.config import Settings
from app.schemas.recognition import RecognitionResponse, VerificationResponse
from app.services.enrollment_service import EnrollmentService
from app.services.recognition_service import RecognitionService
from app.utils.image import decode_image

router = APIRouter(tags=["recognition"])

RecognitionDependency = Annotated[RecognitionService, Depends(get_recognition_service)]
SettingsDependency = Annotated[Settings, Depends(get_settings_dependency)]


@router.post("/recognize", response_model=RecognitionResponse)
def recognize_faces(
    service: RecognitionDependency,
    enrollment_service: Annotated[EnrollmentService, Depends(get_enrollment_service)],
    settings: SettingsDependency,
    image: Annotated[UploadFile, File()],
) -> RecognitionResponse:
    """Identify every face in an image; each match carries the person's name and photo URL."""
    data, _ = read_image_upload(image, settings.max_upload_size_bytes)
    recognitions = service.recognize_faces(decode_image(data))
    names = {record.person_id: record.name for record in enrollment_service.list_faces()}
    return RecognitionResponse(
        match_threshold=service.match_threshold,
        faces=[present_recognition(recognition, names) for recognition in recognitions],
    )


@router.post("/verify", response_model=VerificationResponse)
def verify_faces(
    service: RecognitionDependency,
    settings: SettingsDependency,
    first_image: Annotated[UploadFile, File()],
    second_image: Annotated[UploadFile, File()],
) -> VerificationResponse:
    """1:1 comparison of the largest face in each image."""
    first_data, _ = read_image_upload(first_image, settings.max_upload_size_bytes)
    second_data, _ = read_image_upload(second_image, settings.max_upload_size_bytes)
    result = service.verify_faces(decode_image(first_data), decode_image(second_data))
    return VerificationResponse(
        similarity=round(result.similarity, 4),
        match_threshold=service.match_threshold,
        is_match=result.is_match,
    )

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.api.dependencies import (
    get_check_in_service,
    get_recognition_service,
    get_settings_dependency,
)
from app.api.uploads import read_image_upload
from app.api.v1.presenters import present_check_in, present_photo_check_in
from app.core.config import Settings
from app.schemas.check_in import CheckInListResponse, PhotoCheckInResponse
from app.services.check_in_service import CheckInService
from app.services.recognition_service import RecognitionService
from app.utils.image import decode_image

router = APIRouter(prefix="/check-ins", tags=["check-ins"])

CheckInDependency = Annotated[CheckInService, Depends(get_check_in_service)]
SettingsDependency = Annotated[Settings, Depends(get_settings_dependency)]


@router.get("", response_model=CheckInListResponse)
def list_check_ins(
    service: CheckInDependency,
    settings: SettingsDependency,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
) -> CheckInListResponse:
    """Matched participants, oldest first. Everyone recognized together arrives together.

    This is the TV's poll, so it also keeps camera check-ins on while TV mode is open.
    """
    service.mark_display_active()
    check_ins = service.list_check_ins(after_sequence)
    return CheckInListResponse(
        session_id=service.session_id,
        last_sequence=check_ins[-1].sequence if check_ins else after_sequence,
        display_seconds=settings.tv_display_seconds,
        check_ins=[present_check_in(check_in) for check_in in check_ins],
    )


@router.post("", response_model=PhotoCheckInResponse)
def check_in_photo(
    service: CheckInDependency,
    recognition_service: Annotated[RecognitionService, Depends(get_recognition_service)],
    settings: SettingsDependency,
    image: Annotated[UploadFile, File()],
) -> PhotoCheckInResponse:
    """Check in every registered face in an uploaded photo; they join the TV queue together."""
    data, _ = read_image_upload(image, settings.max_upload_size_bytes)
    recognitions = recognition_service.recognize_faces(decode_image(data))
    return present_photo_check_in(service.record_photo(recognitions))


@router.delete("/{sequence}", status_code=status.HTTP_204_NO_CONTENT)
def remove_check_in(sequence: int, service: CheckInDependency) -> None:
    """Undo one check-in (for testing); the person can check in again right away."""
    service.remove_check_in(sequence)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def reset_check_ins(service: CheckInDependency) -> None:
    """Start a new session: clear the queue and let everyone check in again."""
    service.reset()

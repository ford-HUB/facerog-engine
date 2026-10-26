from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import get_enrollment_service, get_settings_dependency
from app.api.uploads import read_image_upload
from app.api.v1.presenters import present_face, present_summary
from app.core.config import Settings
from app.core.exceptions import FaceNotFoundError
from app.schemas.face import EnrollmentSummaryResponse, FaceListResponse, FaceResponse
from app.services.enrollment_service import EnrollmentService

router = APIRouter(prefix="/faces", tags=["faces"])

EnrollmentDependency = Annotated[EnrollmentService, Depends(get_enrollment_service)]


@router.get("", response_model=FaceListResponse)
def list_faces(service: EnrollmentDependency) -> FaceListResponse:
    faces = [present_face(record) for record in service.list_faces()]
    return FaceListResponse(total=len(faces), faces=faces)


@router.post("", response_model=FaceResponse, status_code=status.HTTP_201_CREATED)
def enroll_face(
    service: EnrollmentDependency,
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    name: Annotated[str, Form(min_length=1, max_length=100)],
    image: Annotated[UploadFile, File()],
) -> FaceResponse:
    """Enroll a photo with exactly one face. It is saved into the dataset folder."""
    data, extension = read_image_upload(image, settings.max_upload_size_bytes)
    return present_face(service.enroll_face(name.strip(), data, extension))


@router.post("/reload", response_model=EnrollmentSummaryResponse)
def reload_dataset(service: EnrollmentDependency) -> EnrollmentSummaryResponse:
    """Re-scan the dataset folder and rebuild the gallery."""
    return present_summary(service.enroll_dataset())


@router.get("/{person_id}", response_model=FaceResponse)
def get_face(person_id: str, service: EnrollmentDependency) -> FaceResponse:
    return present_face(service.get_face(person_id))


@router.get("/{person_id}/photo", response_class=FileResponse)
def get_face_photo(person_id: str, service: EnrollmentDependency) -> FileResponse:
    """The registered picture of an enrolled person."""
    record = service.get_face(person_id)
    if not record.photo_path.is_file():
        raise FaceNotFoundError(f"Photo for '{person_id}' is missing from the dataset.")
    return FileResponse(record.photo_path)


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_face(person_id: str, service: EnrollmentDependency) -> None:
    """Remove the person from the gallery and delete their dataset folder."""
    service.delete_face(person_id)

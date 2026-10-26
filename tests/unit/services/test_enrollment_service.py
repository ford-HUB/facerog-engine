import pytest

from app.api.dependencies import ServiceContainer
from app.core.exceptions import FaceNotDetectedError
from tests.conftest import encode_png, make_face_image


def test_enroll_dataset_builds_four_templates_per_photo(container: ServiceContainer) -> None:
    summary = container.enrollment_service.enroll_dataset()
    assert summary.person_count == 1
    assert summary.template_count == 8  # 2 photos x (original, mirror, dark, low-res)
    record = container.enrollment_service.get_face("red_person")
    assert record.name == "Red Person"


def test_enroll_face_saves_photo_into_dataset(container: ServiceContainer, dataset_dir) -> None:
    record = container.enrollment_service.enroll_face("Blue Person", encode_png(make_face_image("blue")), ".png")
    assert record.person_id == "blue_person"
    assert record.photo_path.is_file()
    assert (dataset_dir / "blue_person" / "info.json").is_file()


def test_enroll_face_without_face_raises(container: ServiceContainer) -> None:
    black = make_face_image("red") * 0
    with pytest.raises(FaceNotDetectedError):
        container.enrollment_service.enroll_face("Nobody", encode_png(black), ".png")


def test_delete_face_removes_record_and_folder(container: ServiceContainer, dataset_dir) -> None:
    container.enrollment_service.enroll_dataset()
    container.enrollment_service.delete_face("red_person")
    assert container.enrollment_service.list_faces() == []
    assert not (dataset_dir / "red_person").exists()

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import CheckInNotFoundError
from app.ml.detector import DetectedFace
from app.ml.head_pose import HeadPose
from app.models.face_record import FaceRecord
from app.services.check_in_service import CheckInService, CheckInSource, PhotoFaceStatus
from app.services.recognition_service import FaceRecognition, Identification
from app.services.track_state import RecognitionEvent


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _open_tv_service(clock: FakeClock | None = None) -> CheckInService:
    """A service whose TV has just polled, so camera check-ins are on."""
    service = CheckInService(cooldown_seconds=60, heartbeat_seconds=5, clock=clock or FakeClock())
    service.mark_display_active()
    return service


def _event(person_id: str, track_id: int = 1) -> RecognitionEvent:
    return RecognitionEvent(
        timestamp=0.0, track_id=track_id, person_id=person_id, name=person_id.title(), similarity=0.6
    )


def test_record_identification_checks_person_in_once_within_cooldown() -> None:
    service = _open_tv_service()
    assert service.record_identification(_event("alice")) is not None
    assert service.record_identification(_event("alice", track_id=2)) is None
    assert [c.person_id for c in service.list_check_ins()] == ["alice"]


def test_record_identification_after_cooldown_checks_in_again() -> None:
    clock = FakeClock()
    service = _open_tv_service(clock)
    service.record_identification(_event("alice"))
    clock.now += 61
    service.mark_display_active()
    assert service.record_identification(_event("alice")).sequence == 2


def test_record_identification_while_tv_closed_checks_no_one_in() -> None:
    service = CheckInService(cooldown_seconds=60, heartbeat_seconds=5, clock=FakeClock())

    assert not service.is_display_active
    assert service.record_identification(_event("alice")) is None
    assert service.list_check_ins() == []


def test_camera_check_ins_stop_when_tv_heartbeat_expires() -> None:
    clock = FakeClock()
    service = _open_tv_service(clock)
    clock.now += 6

    assert service.record_identification(_event("alice")) is None
    service.mark_display_active()
    assert service.record_identification(_event("alice")) is not None


def test_list_check_ins_after_sequence_returns_only_newer() -> None:
    service = _open_tv_service()
    for person_id in ("alice", "bob", "carol"):
        service.record_identification(_event(person_id))
    assert [c.person_id for c in service.list_check_ins(after_sequence=1)] == ["bob", "carol"]


def test_reset_clears_queue_and_changes_session() -> None:
    service = _open_tv_service()
    service.record_identification(_event("alice"))
    session_before = service.session_id

    service.reset()

    assert service.list_check_ins() == []
    assert service.session_id != session_before
    assert service.record_identification(_event("alice")).sequence == 1


def test_remove_check_in_lets_person_check_in_again_immediately() -> None:
    service = _open_tv_service()
    first = service.record_identification(_event("alice"))
    service.record_identification(_event("bob"))

    service.remove_check_in(first.sequence)

    assert [c.person_id for c in service.list_check_ins()] == ["bob"]
    again = service.record_identification(_event("alice"))
    assert again is not None
    assert again.sequence == 3


def test_remove_check_in_unknown_sequence_raises() -> None:
    service = _open_tv_service()
    with pytest.raises(CheckInNotFoundError):
        service.remove_check_in(99)


def _recognition(person_id: str | None, x1: float) -> FaceRecognition:
    person = None
    if person_id is not None:
        person = FaceRecord(
            person_id=person_id,
            name=person_id.title(),
            photo_path=Path("profile.jpg"),
            image_count=1,
            template_count=4,
            embedding_model="fake",
            enrolled_at=datetime(2026, 1, 1),
        )
    detection = DetectedFace(
        bbox=np.array([x1, 0, x1 + 50, 50], dtype=np.float32),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=0.9,
    )
    return FaceRecognition(
        detection=detection,
        head_pose=HeadPose(0.0, 0.0, 0.0),
        is_angled=False,
        identification=Identification(person=person, similarity=0.7, candidates=()),
    )


def test_record_photo_checks_in_matched_faces_left_to_right() -> None:
    service = _open_tv_service()

    photo = [_recognition("bob", 300), _recognition(None, 200), _recognition("alice", 0)]

    results = service.record_photo(photo)

    assert [r.status for r in results] == [
        PhotoFaceStatus.CHECKED_IN,
        PhotoFaceStatus.NOT_REGISTERED,
        PhotoFaceStatus.CHECKED_IN,
    ]
    check_ins = service.list_check_ins()
    assert [c.person_id for c in check_ins] == ["alice", "bob"]
    assert all(c.source is CheckInSource.PHOTO and c.track_id is None for c in check_ins)


def test_record_photo_reports_person_already_checked_in() -> None:
    service = _open_tv_service()
    service.record_identification(_event("alice"))

    photo = [_recognition("alice", 0), _recognition("bob", 100), _recognition("bob", 200)]

    results = service.record_photo(photo)

    assert [r.status for r in results] == [
        PhotoFaceStatus.ALREADY_CHECKED_IN,
        PhotoFaceStatus.CHECKED_IN,
        PhotoFaceStatus.ALREADY_CHECKED_IN,
    ]
    assert [c.person_id for c in service.list_check_ins()] == ["alice", "bob"]

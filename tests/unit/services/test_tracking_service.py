from app.api.dependencies import ServiceContainer, wire_services
from app.core.config import Settings
from app.ml.face_analyzer import FaceAnalyzer
from app.services.track_state import TrackStatus
from tests.conftest import make_face_image


def test_matched_track_keeps_name_when_recheck_is_weaker(
    settings: Settings, face_analyzer: FaceAnalyzer
) -> None:
    eager_settings = settings.model_copy(update={"reverify_interval_frames": 1})
    container = wire_services(eager_settings, face_analyzer, face_analyzer)
    container.enrollment_service.enroll_dataset()
    tracking = container.tracking_service
    tracking.process_frame(make_face_image("red", size=400))

    for _ in range(3):
        tracking.process_frame(make_face_image("green", size=400))

    track = tracking.get_snapshot().tracks[0]
    assert track.recognition_count == 4
    assert track.status == TrackStatus.MATCHED
    assert track.identification.person.name == "Red Person"


def test_process_frame_identifies_enrolled_face_and_reuses_identity(container: ServiceContainer) -> None:
    container.enrollment_service.enroll_dataset()
    tracking = container.tracking_service

    for _ in range(4):
        tracking.process_frame(make_face_image("red", size=400))

    snapshot = tracking.get_snapshot()
    assert len(snapshot.tracks) == 1
    track = snapshot.tracks[0]
    assert track.status == TrackStatus.MATCHED
    assert track.identification.person.name == "Red Person"
    assert track.recognition_count == 1
    assert tracking.get_zoom_jpeg(track.track_id).startswith(b"\xff\xd8")
    assert [event.name for event in tracking.get_events()] == ["Red Person"]


def test_process_frame_unknown_face_is_marked_unknown(container: ServiceContainer) -> None:
    container.enrollment_service.enroll_dataset()
    tracking = container.tracking_service

    for _ in range(3):
        tracking.process_frame(make_face_image("green", size=400))

    assert tracking.get_snapshot().tracks[0].status == TrackStatus.UNKNOWN
    assert tracking.get_events() == []

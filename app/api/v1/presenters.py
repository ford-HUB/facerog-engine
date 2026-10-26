from app.models.face_record import FaceRecord
from app.schemas.check_in import CheckInResponse, PhotoCheckInResponse, PhotoFaceCheckInResponse
from app.schemas.face import EnrollmentSummaryResponse, FaceResponse
from app.schemas.recognition import (
    BoundingBox,
    CandidateMatch,
    HeadPoseEstimate,
    PersonMatch,
    RecognizedFace,
)
from app.schemas.tracking import RecognitionEventResponse, TrackingSnapshotResponse, TrackResponse
from app.services.check_in_service import CheckIn, PhotoFaceCheckIn
from app.services.enrollment_service import EnrollmentSummary
from app.services.recognition_service import FaceRecognition, Identification
from app.services.track_state import RecognitionEvent, TrackingSnapshot, TrackView

API_PREFIX = "/api/v1"


def photo_url(person_id: str) -> str:
    return f"{API_PREFIX}/faces/{person_id}/photo"


def present_face(record: FaceRecord) -> FaceResponse:
    return FaceResponse(
        person_id=record.person_id,
        name=record.name,
        photo_url=photo_url(record.person_id),
        image_count=record.image_count,
        template_count=record.template_count,
        embedding_model=record.embedding_model,
        enrolled_at=record.enrolled_at,
    )


def present_summary(summary: EnrollmentSummary) -> EnrollmentSummaryResponse:
    return EnrollmentSummaryResponse(
        person_count=summary.person_count,
        template_count=summary.template_count,
        skipped_images=list(summary.skipped_images),
    )


def present_person(record: FaceRecord) -> PersonMatch:
    return PersonMatch(person_id=record.person_id, name=record.name, photo_url=photo_url(record.person_id))


def present_head_pose(source: FaceRecognition | TrackView) -> HeadPoseEstimate:
    pose = source.head_pose
    return HeadPoseEstimate(
        yaw_degrees=round(pose.yaw_degrees, 1),
        pitch_degrees=round(pose.pitch_degrees, 1),
        roll_degrees=round(pose.roll_degrees, 1),
        is_angled=source.is_angled,
    )


def present_recognition(recognition: FaceRecognition, names: dict[str, str]) -> RecognizedFace:
    identification = recognition.identification
    x1, y1, x2, y2 = (float(v) for v in recognition.detection.bbox)
    return RecognizedFace(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        detection_score=round(recognition.detection.score, 4),
        head_pose=present_head_pose(recognition),
        is_match=identification.is_match,
        similarity=round(identification.similarity, 4),
        person=present_person(identification.person) if identification.person else None,
        candidates=_present_candidates(identification, names),
    )


def present_snapshot(snapshot: TrackingSnapshot) -> TrackingSnapshotResponse:
    return TrackingSnapshotResponse(
        is_running=snapshot.is_running,
        is_source_open=snapshot.is_source_open,
        video_source=snapshot.video_source,
        frame_index=snapshot.frame_index,
        fps=snapshot.fps,
        frame_width=snapshot.frame_width,
        frame_height=snapshot.frame_height,
        match_threshold=snapshot.match_threshold,
        tracks=[_present_track(view) for view in snapshot.tracks],
    )


def present_event(event: RecognitionEvent) -> RecognitionEventResponse:
    return RecognitionEventResponse(
        timestamp=event.timestamp,
        track_id=event.track_id,
        person=PersonMatch(person_id=event.person_id, name=event.name, photo_url=photo_url(event.person_id)),
        similarity=round(event.similarity, 4),
    )


def present_check_in(check_in: CheckIn) -> CheckInResponse:
    return CheckInResponse(
        sequence=check_in.sequence,
        person=PersonMatch(
            person_id=check_in.person_id, name=check_in.name, photo_url=photo_url(check_in.person_id)
        ),
        similarity=round(check_in.similarity, 4),
        source=check_in.source.value,
        track_id=check_in.track_id,
        checked_in_at=check_in.checked_in_at,
    )


def present_photo_check_in(results: list[PhotoFaceCheckIn]) -> PhotoCheckInResponse:
    return PhotoCheckInResponse(
        checked_in_count=sum(1 for result in results if result.check_in is not None),
        faces=[_present_photo_face(result) for result in results],
    )


def _present_photo_face(result: PhotoFaceCheckIn) -> PhotoFaceCheckInResponse:
    identification = result.recognition.identification
    x1, y1, x2, y2 = (float(v) for v in result.recognition.detection.bbox)
    return PhotoFaceCheckInResponse(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        status=result.status.value,
        similarity=round(identification.similarity, 4),
        person=present_person(identification.person) if identification.person else None,
        check_in=present_check_in(result.check_in) if result.check_in else None,
    )


def _present_track(view: TrackView) -> TrackResponse:
    identification = view.identification
    x1, y1, x2, y2 = view.bbox
    return TrackResponse(
        track_id=view.track_id,
        status=view.status.value,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        detection_score=round(view.detection_score, 4),
        head_pose=present_head_pose(view),
        similarity=round(identification.similarity, 4) if identification else None,
        person=present_person(identification.person) if identification and identification.person else None,
        zoom_url=f"{API_PREFIX}/stream/tracks/{view.track_id}/zoom",
        trajectory=[(round(x, 4), round(y, 4)) for x, y in view.trajectory],
        seconds_tracked=round(view.seconds_tracked, 1),
        recognition_count=view.recognition_count,
    )


def _present_candidates(identification: Identification, names: dict[str, str]) -> list[CandidateMatch]:
    return [
        CandidateMatch(
            person_id=candidate.person_id,
            name=names.get(candidate.person_id, candidate.person_id),
            similarity=round(candidate.similarity, 4),
        )
        for candidate in identification.candidates
    ]

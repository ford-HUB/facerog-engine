from pydantic import BaseModel

from app.schemas.recognition import BoundingBox, HeadPoseEstimate, PersonMatch


class TrackResponse(BaseModel):
    track_id: int
    status: str
    bbox: BoundingBox
    detection_score: float
    head_pose: HeadPoseEstimate
    similarity: float | None
    person: PersonMatch | None
    zoom_url: str
    trajectory: list[tuple[float, float]]
    seconds_tracked: float
    recognition_count: int


class TrackingSnapshotResponse(BaseModel):
    is_running: bool
    is_source_open: bool
    video_source: str
    frame_index: int
    fps: float
    frame_width: int
    frame_height: int
    match_threshold: float
    tracks: list[TrackResponse]


class RecognitionEventResponse(BaseModel):
    timestamp: float
    track_id: int
    person: PersonMatch
    similarity: float


class StartStreamRequest(BaseModel):
    source: str | None = None

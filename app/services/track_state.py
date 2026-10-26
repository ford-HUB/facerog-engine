"""Per-track memory used by the tracking service: where a face has been and who it is."""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.ml.detector import DetectedFace
from app.ml.embedder import Embedding
from app.ml.head_pose import HeadPose, estimate_head_pose
from app.ml.matcher import aggregate_embeddings
from app.services.recognition_service import Identification


class TrackStatus(str, Enum):
    IDENTIFYING = "identifying"
    MATCHED = "matched"
    UNKNOWN = "unknown"


@dataclass
class TrackState:
    """Mutable state for one ByteTrack id. Owned by the tracking thread only."""

    track_id: int
    trajectory_length: int
    embedding_history: int
    first_seen_at: float = field(default_factory=time.time)
    last_seen_frame: int = 0
    last_recognized_frame: int | None = None
    recognition_count: int = 0
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    detection: DetectedFace | None = None
    head_pose: HeadPose = HeadPose(0.0, 0.0, 0.0)
    identification: Identification | None = None
    trajectory: deque[tuple[float, float]] = field(init=False)
    _embeddings: deque[Embedding] = field(init=False)
    _weights: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.trajectory = deque(maxlen=self.trajectory_length)
        self._embeddings = deque(maxlen=self.embedding_history)
        self._weights = deque(maxlen=self.embedding_history)

    @property
    def status(self) -> TrackStatus:
        if self.identification is None:
            return TrackStatus.IDENTIFYING
        return TrackStatus.MATCHED if self.identification.is_match else TrackStatus.UNKNOWN

    def observe(self, detection: DetectedFace, bbox: np.ndarray, frame_shape: tuple[int, ...], frame_index: int) -> None:
        """Record this frame's detection; trajectory points are normalized to [0, 1]."""
        height, width = frame_shape[:2]
        self.detection = detection
        self.bbox = bbox.astype(np.float32)
        self.head_pose = estimate_head_pose(detection.landmarks)
        self.last_seen_frame = frame_index
        center_x = float((bbox[0] + bbox[2]) / 2 / width)
        center_y = float((bbox[1] + bbox[3]) / 2 / height)
        self.trajectory.append((min(max(center_x, 0.0), 1.0), min(max(center_y, 0.0), 1.0)))

    def add_embedding(self, embedding: Embedding, weight: float) -> Embedding:
        """Add a frame's embedding and return the track's aggregated identity vector."""
        self._embeddings.append(embedding)
        self._weights.append(weight)
        return aggregate_embeddings(list(self._embeddings), list(self._weights))

    def is_recognition_due(self, frame_index: int, retry_interval: int, reverify_interval: int) -> bool:
        """Matched tracks are only re-checked occasionally; others keep trying."""
        if self.last_recognized_frame is None:
            return True
        interval = reverify_interval if self.status == TrackStatus.MATCHED else retry_interval
        return frame_index - self.last_recognized_frame >= interval


@dataclass(frozen=True)
class TrackView:
    """Read-only copy of a track for the API."""

    track_id: int
    bbox: tuple[float, float, float, float]
    detection_score: float
    head_pose: HeadPose
    is_angled: bool
    status: TrackStatus
    identification: Identification | None
    trajectory: tuple[tuple[float, float], ...]
    seconds_tracked: float
    recognition_count: int


@dataclass(frozen=True)
class RecognitionEvent:
    """Emitted when a track is first identified as someone (or switches identity)."""

    timestamp: float
    track_id: int
    person_id: str
    name: str
    similarity: float


@dataclass(frozen=True)
class TrackingSnapshot:
    is_running: bool
    is_source_open: bool
    video_source: str
    frame_index: int
    fps: float
    frame_width: int
    frame_height: int
    match_threshold: float
    tracks: tuple[TrackView, ...]

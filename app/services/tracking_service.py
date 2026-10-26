import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.exceptions import StreamUnavailableError, TrackNotFoundError
from app.ml.detector import DetectedFace
from app.ml.face_analyzer import FaceAnalyzer
from app.ml.tracker import ByteTracker, Track, TrackerConfig
from app.services.recognition_service import Identification, RecognitionService
from app.services.track_state import (
    RecognitionEvent,
    TrackingSnapshot,
    TrackState,
    TrackStatus,
    TrackView,
)
from app.utils import drawing
from app.utils.image import encode_jpeg, resize_to_width
from app.utils.video_stream import FrameSource

logger = logging.getLogger(__name__)

IDLE_WAIT_SECONDS = 0.005
FPS_SMOOTHING = 0.9
MAX_EVENTS = 50


@dataclass(frozen=True)
class TrackingConfig:
    frame_width: int
    jpeg_quality: int
    tracker: TrackerConfig
    trajectory_length: int
    embedding_history: int
    recognition_interval_frames: int
    reverify_interval_frames: int
    angled_face_threshold_degrees: float
    zoom_size_pixels: int


class TrackingService:
    """Runs the live pipeline on a background thread.

    Per frame: detect faces (SCRFD) -> associate them with tracks (ByteTrack) ->
    run ArcFace only for tracks whose identity is unknown or due for re-verification.
    Publishes an annotated JPEG, per-track zoom images and a JSON-friendly snapshot.
    """

    def __init__(
        self,
        analyzer: FaceAnalyzer,
        recognition_service: RecognitionService,
        source_factory: Callable[[str], FrameSource],
        video_source: str,
        config: TrackingConfig,
        on_identified: Callable[[RecognitionEvent], object] | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._recognition_service = recognition_service
        self._on_identified = on_identified
        self._source_factory = source_factory
        self._video_source = video_source
        self._config = config
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._reset_pipeline()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, video_source: str | None = None) -> None:
        """Start (or restart with a new source) the capture and tracking threads."""
        if video_source is not None and video_source != self._video_source:
            self.stop()
            self._video_source = video_source
        if self.is_running:
            return
        self._reset_pipeline()
        self._source = self._source_factory(self._video_source)
        self._source.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="face-tracking", daemon=True)
        self._thread.start()
        logger.info("Tracking started on source %r", self._video_source)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._source:
            self._source.stop()
        self._thread = None
        self._source = None

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            jpeg = self._latest_jpeg
        if jpeg is None:
            raise StreamUnavailableError("No frame has been processed yet.")
        return jpeg

    def get_zoom_jpeg(self, track_id: int) -> bytes:
        with self._lock:
            jpeg = self._zoom_jpegs.get(track_id)
        if jpeg is None:
            raise TrackNotFoundError(f"No active track with id {track_id}.")
        return jpeg

    def get_snapshot(self) -> TrackingSnapshot:
        with self._lock:
            return TrackingSnapshot(
                is_running=self.is_running,
                is_source_open=bool(self._source and self._source.is_open),
                video_source=self._video_source,
                frame_index=self._frame_index,
                fps=round(self._fps, 1),
                frame_width=self._frame_size[0],
                frame_height=self._frame_size[1],
                match_threshold=self._recognition_service.match_threshold,
                tracks=self._track_views,
            )

    def get_events(self) -> list[RecognitionEvent]:
        with self._lock:
            return list(reversed(self._events))

    def process_frame(self, frame: np.ndarray) -> None:
        """Run one frame through the pipeline. Called by the worker thread (and by tests)."""
        started_at = time.perf_counter()
        frame = resize_to_width(frame, self._config.frame_width)
        detections = self._analyzer.detect_faces(frame)
        tracks = self._update_tracker(detections)
        self._frame_index += 1
        active_states = self._observe_tracks(frame, tracks, detections)
        self._recognize_due_tracks(frame, active_states)
        self._forget_expired_tracks()
        self._publish(frame, active_states, time.perf_counter() - started_at)

    def _reset_pipeline(self) -> None:
        self._tracker = ByteTracker(self._config.tracker)
        self._states: dict[int, TrackState] = {}
        self._frame_index = 0
        self._fps = 0.0
        self._frame_size = (0, 0)
        self._latest_jpeg: bytes | None = None
        self._zoom_jpegs: dict[int, bytes] = {}
        self._track_views: tuple[TrackView, ...] = ()
        self._events: deque[RecognitionEvent] = deque(maxlen=MAX_EVENTS)

    def _run(self) -> None:
        last_frame_number = 0
        while not self._stop_event.is_set():
            assert self._source is not None
            frame_number, frame = self._source.read_latest()
            if frame is None or frame_number == last_frame_number:
                self._stop_event.wait(IDLE_WAIT_SECONDS)
                continue
            last_frame_number = frame_number
            try:
                self.process_frame(frame)
            except (ValueError, RuntimeError, cv2.error) as err:
                # Keep the stream alive through a single bad frame; log it for diagnosis.
                logger.exception("Frame %d failed: %s", frame_number, err)

    def _update_tracker(self, detections: list[DetectedFace]) -> list[Track]:
        bboxes = np.array([face.bbox for face in detections], dtype=np.float32).reshape(-1, 4)
        scores = np.array([face.score for face in detections], dtype=np.float32)
        return self._tracker.update(bboxes, scores)

    def _observe_tracks(
        self, frame: np.ndarray, tracks: list[Track], detections: list[DetectedFace]
    ) -> list[TrackState]:
        active = []
        for track in tracks:
            state = self._states.get(track.track_id)
            if state is None:
                state = TrackState(
                    track.track_id, self._config.trajectory_length, self._config.embedding_history
                )
                self._states[track.track_id] = state
            state.observe(detections[track.detection_index], track.bbox, frame.shape, self._frame_index)
            active.append(state)
        return active

    def _recognize_due_tracks(self, frame: np.ndarray, states: list[TrackState]) -> None:
        due = [
            state for state in states
            if state.detection is not None
            and state.detection.score >= self._config.tracker.high_threshold
            and state.is_recognition_due(
                self._frame_index,
                self._config.recognition_interval_frames,
                self._config.reverify_interval_frames,
            )
        ]
        if not due:
            return
        analyzed = self._analyzer.embed_detections(frame, [state.detection for state in due])
        for state, face in zip(due, analyzed, strict=True):
            weight = face.head_pose.frontal_score * face.detection.score
            track_embedding = state.add_embedding(face.embedding, weight)
            previous = state.identification
            current = self._recognition_service.identify_embedding(track_embedding)
            state.identification = _keep_established_identity(previous, current)
            state.last_recognized_frame = self._frame_index
            state.recognition_count += 1
            self._publish_identification(state, previous)

    def _publish_identification(self, state: TrackState, previous: Identification | None) -> None:
        """Log new identities, and report every match to the listener (it applies its own cooldown)."""
        current = state.identification
        if current is None or current.person is None:
            return
        person = current.person
        event = RecognitionEvent(
            time.time(), state.track_id, person.person_id, person.name, current.similarity
        )
        is_same_person = previous is not None and previous.person is not None and (
            previous.person.person_id == person.person_id
        )
        if not is_same_person:
            with self._lock:
                self._events.append(event)
            logger.info(
                "Track %d identified as %s (%.3f)", state.track_id, person.person_id, current.similarity
            )
        if self._on_identified is not None:
            self._on_identified(event)

    def _forget_expired_tracks(self) -> None:
        horizon = self._config.tracker.buffer_frames
        expired = [tid for tid, s in self._states.items() if self._frame_index - s.last_seen_frame > horizon]
        for track_id in expired:
            del self._states[track_id]

    def _publish(self, frame: np.ndarray, states: list[TrackState], elapsed_seconds: float) -> None:
        zooms = {state.track_id: self._render_zoom(frame, state) for state in states}
        annotated = frame.copy()
        for state in states:
            drawing.draw_track_box(annotated, state.bbox, _track_label(state), _track_color(state))
        angled = [zooms[s.track_id] for s in states if s.head_pose.is_angled(self._config.angled_face_threshold_degrees)]
        drawing.draw_zoom_insets(annotated, angled)
        jpeg = encode_jpeg(annotated, self._config.jpeg_quality)
        zoom_jpegs = {track_id: encode_jpeg(zoom) for track_id, zoom in zooms.items()}
        views = tuple(self._to_view(state) for state in states)
        instant_fps = 1.0 / max(elapsed_seconds, 1e-6)
        with self._lock:
            self._latest_jpeg = jpeg
            self._zoom_jpegs = zoom_jpegs
            self._track_views = views
            self._frame_size = (frame.shape[1], frame.shape[0])
            self._fps = instant_fps if self._fps == 0 else FPS_SMOOTHING * self._fps + (1 - FPS_SMOOTHING) * instant_fps

    def _render_zoom(self, frame: np.ndarray, state: TrackState) -> np.ndarray:
        assert state.detection is not None
        pose = state.head_pose
        caption = f"#{state.track_id} yaw {pose.yaw_degrees:+.0f} pitch {pose.pitch_degrees:+.0f}"
        return drawing.render_zoom(
            frame, state.bbox, state.detection.landmarks, self._config.zoom_size_pixels, caption
        )

    def _to_view(self, state: TrackState) -> TrackView:
        x1, y1, x2, y2 = (float(v) for v in state.bbox)
        return TrackView(
            track_id=state.track_id,
            bbox=(x1, y1, x2, y2),
            detection_score=state.detection.score if state.detection else 0.0,
            head_pose=state.head_pose,
            is_angled=state.head_pose.is_angled(self._config.angled_face_threshold_degrees),
            status=state.status,
            identification=state.identification,
            trajectory=tuple(state.trajectory),
            seconds_tracked=time.time() - state.first_seen_at,
            recognition_count=state.recognition_count,
        )


def _keep_established_identity(
    previous: Identification | None, current: Identification
) -> Identification:
    """A matched track keeps its name through angled, dim or blurred frames.

    ByteTrack guarantees the track is still the same face, so a weaker re-check is not
    evidence of a different person. The name only changes if another enrolled person matches.
    """
    if previous is not None and previous.is_match and not current.is_match:
        return previous
    return current


def _track_label(state: TrackState) -> str:
    identification = state.identification
    if identification is None:
        return f"#{state.track_id} identifying..."
    name = identification.person.name if identification.person else "Unknown"
    return f"#{state.track_id} {name} {identification.similarity:.2f}"


def _track_color(state: TrackState) -> tuple[int, int, int]:
    return {
        TrackStatus.MATCHED: drawing.MATCHED_COLOR,
        TrackStatus.UNKNOWN: drawing.UNKNOWN_COLOR,
        TrackStatus.IDENTIFYING: drawing.PENDING_COLOR,
    }[state.status]

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.ml.detector import ScrfdDetector
from app.ml.embedder import ArcFaceEmbedder
from app.ml.face_analyzer import FaceAnalyzer
from app.ml.onnx_session import create_session
from app.ml.tracker import TrackerConfig
from app.repositories.face_repository import FaceRepository
from app.repositories.vector_index import InMemoryVectorIndex
from app.services.check_in_service import CheckInService
from app.services.enrollment_service import EnrollmentService
from app.services.recognition_service import RecognitionService
from app.services.tracking_service import TrackingConfig, TrackingService
from app.utils.video_stream import OpenCvVideoStream


@dataclass(frozen=True)
class ServiceContainer:
    settings: Settings
    enrollment_service: EnrollmentService
    recognition_service: RecognitionService
    tracking_service: TrackingService
    check_in_service: CheckInService
    execution_providers: tuple[str, ...]


def build_container(settings: Settings) -> ServiceContainer:
    """Load models once and wire every service. Raises ModelLoadError on missing weights."""
    detector_session = create_session(settings.detector_model_path, settings.use_gpu)
    embedder_session = create_session(settings.embedder_model_path, settings.use_gpu)
    embedder = ArcFaceEmbedder(embedder_session, model_name=settings.embedder_model_path.stem)

    def make_analyzer(confidence_threshold: float) -> FaceAnalyzer:
        detector = ScrfdDetector(
            detector_session,
            input_size=settings.detection_input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=settings.detection_nms_threshold,
        )
        return FaceAnalyzer(detector, embedder, settings.min_face_size_pixels)

    # ByteTrack needs low-confidence boxes too, so the live pipeline detects with a lower cutoff.
    still_analyzer = make_analyzer(settings.detection_confidence_threshold)
    tracking_analyzer = make_analyzer(settings.track_low_threshold)
    return wire_services(
        settings,
        still_analyzer,
        tracking_analyzer,
        execution_providers=tuple(embedder_session.get_providers()),
    )


def wire_services(
    settings: Settings,
    still_analyzer: FaceAnalyzer,
    tracking_analyzer: FaceAnalyzer,
    execution_providers: tuple[str, ...] = (),
) -> ServiceContainer:
    """Build services around already-constructed analyzers (also used by tests with fakes)."""
    repository = FaceRepository(settings.dataset_dir)
    vector_index = InMemoryVectorIndex(still_analyzer.embedding_model)
    recognition_service = RecognitionService(
        still_analyzer,
        repository,
        vector_index,
        match_threshold=settings.match_threshold,
        top_k=settings.top_k,
        angled_face_threshold_degrees=settings.angled_face_threshold_degrees,
    )
    check_in_service = CheckInService(
        cooldown_seconds=settings.check_in_cooldown_minutes * 60,
        heartbeat_seconds=settings.tv_heartbeat_seconds,
    )
    tracking_service = TrackingService(
        tracking_analyzer,
        recognition_service,
        source_factory=OpenCvVideoStream,
        video_source=settings.video_source,
        config=_tracking_config(settings),
        on_identified=check_in_service.record_identification,
    )
    return ServiceContainer(
        settings=settings,
        enrollment_service=EnrollmentService(still_analyzer, repository, vector_index),
        recognition_service=recognition_service,
        tracking_service=tracking_service,
        check_in_service=check_in_service,
        execution_providers=execution_providers,
    )


def _tracking_config(settings: Settings) -> TrackingConfig:
    return TrackingConfig(
        frame_width=settings.video_frame_width,
        jpeg_quality=settings.stream_jpeg_quality,
        tracker=TrackerConfig(
            high_threshold=settings.track_high_threshold,
            low_threshold=settings.track_low_threshold,
            new_track_threshold=settings.track_new_threshold,
            buffer_frames=settings.track_buffer_frames,
        ),
        trajectory_length=settings.trajectory_length,
        embedding_history=settings.track_embedding_history,
        recognition_interval_frames=settings.recognition_interval_frames,
        reverify_interval_frames=settings.reverify_interval_frames,
        angled_face_threshold_degrees=settings.angled_face_threshold_degrees,
        zoom_size_pixels=settings.zoom_size_pixels,
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_settings_dependency(request: Request) -> Settings:
    return get_container(request).settings


def get_enrollment_service(request: Request) -> EnrollmentService:
    return get_container(request).enrollment_service


def get_recognition_service(request: Request) -> RecognitionService:
    return get_container(request).recognition_service


def get_tracking_service(request: Request) -> TrackingService:
    return get_container(request).tracking_service


def get_check_in_service(request: Request) -> CheckInService:
    return get_container(request).check_in_service

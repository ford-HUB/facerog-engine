from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Engine configuration. Every field maps to an `FRE_<FIELD_NAME>` env var."""

    model_config = SettingsConfigDict(
        env_prefix="FRE_", env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    model_dir: Path = Path("app/ml/weights")
    detector_model_file: str = "det_10g.onnx"
    embedder_model_file: str = "w600k_r50.onnx"
    use_gpu: bool = True

    dataset_dir: Path = Path("datasets/faces")

    detection_input_size: int = 640
    detection_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_nms_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    min_face_size_pixels: int = 24

    match_threshold: float = Field(default=0.40, ge=-1.0, le=1.0)
    top_k: int = 3

    video_source: str = "0"
    video_autostart: bool = True
    video_frame_width: int = 960
    stream_jpeg_quality: int = Field(default=80, ge=10, le=100)
    stream_max_fps: float = 25.0

    track_high_threshold: float = 0.6
    track_low_threshold: float = 0.2
    track_new_threshold: float = 0.7
    track_buffer_frames: int = 30
    trajectory_length: int = 60

    recognition_interval_frames: int = 5
    reverify_interval_frames: int = 90
    track_embedding_history: int = 10
    angled_face_threshold_degrees: float = 15.0
    zoom_size_pixels: int = 256

    check_in_cooldown_minutes: float = Field(default=60.0, ge=0.0)
    tv_display_seconds: float = Field(default=3.0, gt=0.0)
    tv_heartbeat_seconds: float = Field(default=5.0, gt=0.0)

    max_upload_size_bytes: int = 10 * 1024 * 1024

    @field_validator("model_dir", "dataset_dir")
    @classmethod
    def _resolve_from_project_root(cls, path: Path) -> Path:
        """Relative paths are anchored at the project root, not the shell's current directory."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def detector_model_path(self) -> Path:
        return self.model_dir / self.detector_model_file

    @property
    def embedder_model_path(self) -> Path:
        return self.model_dir / self.embedder_model_file


@lru_cache
def get_settings() -> Settings:
    return Settings()

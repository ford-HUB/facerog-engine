from pathlib import Path

import cv2
import numpy as np
import pytest

from app.api.dependencies import ServiceContainer, wire_services
from app.core.config import Settings
from app.ml.detector import DetectedFace
from app.ml.embedder import Embedding, l2_normalize
from app.ml.face_analyzer import FaceAnalyzer

EMBEDDING_DIMENSION = 512
FACE_COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}


class FakeDetector:
    """Finds one face in the centre of any non-black image."""

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        if image.max() == 0:
            return []
        height, width = image.shape[:2]
        x1, y1, x2, y2 = width * 0.3, height * 0.3, width * 0.7, height * 0.7
        face_w, face_h = x2 - x1, y2 - y1
        landmarks = np.array(
            [
                [x1 + 0.3 * face_w, y1 + 0.4 * face_h],
                [x1 + 0.7 * face_w, y1 + 0.4 * face_h],
                [x1 + 0.5 * face_w, y1 + 0.6 * face_h],
                [x1 + 0.35 * face_w, y1 + 0.8 * face_h],
                [x1 + 0.65 * face_w, y1 + 0.8 * face_h],
            ],
            dtype=np.float32,
        )
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        return [DetectedFace(bbox=bbox, landmarks=landmarks, score=0.95)]


class FakeEmbedder:
    """Embeds a face as its mean BGR colour, so same-colour images are the same 'person'."""

    model_name = "fake_embedder"

    def embed_faces(self, aligned_faces: list[np.ndarray]) -> list[Embedding]:
        embeddings = []
        for face in aligned_faces:
            vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
            vector[:3] = face.reshape(-1, 3).mean(axis=0) + 1.0
            embeddings.append(l2_normalize(vector))
        return embeddings


def make_face_image(color_name: str, size: int = 200) -> np.ndarray:
    return np.full((size, size, 3), FACE_COLORS[color_name], dtype=np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A dataset with one enrolled person, 'Red Person', who has two photos."""
    person_dir = tmp_path / "faces" / "red_person"
    person_dir.mkdir(parents=True)
    for name in ("01.png", "02.png"):
        cv2.imwrite(str(person_dir / name), make_face_image("red"))
    return tmp_path / "faces"


@pytest.fixture
def settings(dataset_dir: Path) -> Settings:
    return Settings(dataset_dir=dataset_dir, video_autostart=False, match_threshold=0.9, _env_file=None)


@pytest.fixture
def face_analyzer() -> FaceAnalyzer:
    return FaceAnalyzer(FakeDetector(), FakeEmbedder(), min_face_size_pixels=10)


@pytest.fixture
def container(settings: Settings, face_analyzer: FaceAnalyzer) -> ServiceContainer:
    return wire_services(settings, face_analyzer, face_analyzer, execution_providers=("CPUExecutionProvider",))

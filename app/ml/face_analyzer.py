from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.aligner import align_face
from app.ml.augmentation import darken, simulate_low_resolution
from app.ml.detector import DetectedFace, FaceDetector
from app.ml.embedder import Embedding, FaceEmbedder, l2_normalize
from app.ml.head_pose import HeadPose, estimate_head_pose


@dataclass(frozen=True)
class AnalyzedFace:
    detection: DetectedFace
    embedding: Embedding
    head_pose: HeadPose


class FaceAnalyzer:
    """Chains detection, alignment and embedding. Images are BGR uint8."""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, min_face_size_pixels: int) -> None:
        self._detector = detector
        self._embedder = embedder
        self._min_face_size_pixels = min_face_size_pixels

    @property
    def embedding_model(self) -> str:
        return self._embedder.model_name

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces large enough to recognize, highest confidence first."""
        return [face for face in self._detector.detect(image) if self._is_large_enough(face)]

    def embed_detections(self, image: np.ndarray, detections: list[DetectedFace]) -> list[AnalyzedFace]:
        """Embed each face as the average of itself and its mirror image.

        Mirror averaging makes a face turned slightly left and one turned slightly right
        land closer to the same frontal identity, so a single frontal enrollment photo
        still matches angled live faces.
        """
        if not detections:
            return []
        aligned_faces = [align_face(image, face.landmarks) for face in detections]
        mirrored_faces = [cv2.flip(face, 1) for face in aligned_faces]
        raw = self._embedder.embed_faces(aligned_faces + mirrored_faces)
        count = len(aligned_faces)
        embeddings = [l2_normalize(a + b) for a, b in zip(raw[:count], raw[count:], strict=True)]
        return [
            AnalyzedFace(face, embedding, estimate_head_pose(face.landmarks))
            for face, embedding in zip(detections, embeddings, strict=True)
        ]

    def analyze(self, image: np.ndarray) -> list[AnalyzedFace]:
        return self.embed_detections(image, self.detect_faces(image))

    def embed_enrollment_templates(self, image: np.ndarray, detection: DetectedFace) -> list[Embedding]:
        """Turn one enrollment photo into several templates, so one photo per person is enough.

        Templates: the face as taken, mirrored (head turned the other way), darkened
        (dim or backlit webcam) and low resolution (face far from the camera).
        """
        aligned = align_face(image, detection.landmarks)
        variants = [aligned, cv2.flip(aligned, 1), darken(aligned), simulate_low_resolution(aligned)]
        return self._embedder.embed_faces(variants)

    def _is_large_enough(self, face: DetectedFace) -> bool:
        x1, y1, x2, y2 = face.bbox
        return min(x2 - x1, y2 - y1) >= self._min_face_size_pixels

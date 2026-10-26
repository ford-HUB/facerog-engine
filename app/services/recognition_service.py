from dataclasses import dataclass

import numpy as np

from app.core.exceptions import FaceNotDetectedError
from app.ml.detector import DetectedFace
from app.ml.embedder import Embedding
from app.ml.face_analyzer import FaceAnalyzer
from app.ml.head_pose import HeadPose
from app.ml.matcher import Candidate, compute_similarity, is_match, rank_candidates
from app.models.face_record import FaceRecord
from app.repositories.face_repository import FaceRepository
from app.repositories.vector_index import InMemoryVectorIndex


@dataclass(frozen=True)
class Identification:
    """Who a face belongs to. person is None when nobody clears the match threshold."""

    person: FaceRecord | None
    similarity: float
    candidates: tuple[Candidate, ...]

    @property
    def is_match(self) -> bool:
        return self.person is not None


@dataclass(frozen=True)
class FaceRecognition:
    detection: DetectedFace
    head_pose: HeadPose
    is_angled: bool
    identification: Identification


@dataclass(frozen=True)
class VerificationResult:
    similarity: float
    is_match: bool


class RecognitionService:
    """1:N recognition and 1:1 verification against the enrolled gallery."""

    def __init__(
        self,
        analyzer: FaceAnalyzer,
        repository: FaceRepository,
        vector_index: InMemoryVectorIndex,
        match_threshold: float,
        top_k: int,
        angled_face_threshold_degrees: float,
    ) -> None:
        self._analyzer = analyzer
        self._repository = repository
        self._vector_index = vector_index
        self._match_threshold = match_threshold
        self._top_k = top_k
        self._angled_face_threshold_degrees = angled_face_threshold_degrees

    @property
    def match_threshold(self) -> float:
        return self._match_threshold

    def recognize_faces(self, image: np.ndarray) -> list[FaceRecognition]:
        faces = self._analyzer.analyze(image)
        if not faces:
            raise FaceNotDetectedError()
        return [
            FaceRecognition(
                detection=face.detection,
                head_pose=face.head_pose,
                is_angled=face.head_pose.is_angled(self._angled_face_threshold_degrees),
                identification=self.identify_embedding(face.embedding),
            )
            for face in faces
        ]

    def identify_embedding(self, embedding: Embedding) -> Identification:
        gallery = self._vector_index.snapshot()
        candidates = rank_candidates(embedding, gallery.embeddings, gallery.person_ids, self._top_k)
        if not candidates:
            return Identification(person=None, similarity=0.0, candidates=())
        best = candidates[0]
        person = (
            self._repository.find_record(best.person_id)
            if is_match(best.similarity, self._match_threshold)
            else None
        )
        return Identification(person=person, similarity=best.similarity, candidates=tuple(candidates))

    def verify_faces(self, first_image: np.ndarray, second_image: np.ndarray) -> VerificationResult:
        """1:1 check: are the largest faces in the two images the same person?"""
        first = self._largest_face_embedding(first_image)
        second = self._largest_face_embedding(second_image)
        similarity = compute_similarity(first, second)
        return VerificationResult(similarity, is_match(similarity, self._match_threshold))

    def _largest_face_embedding(self, image: np.ndarray) -> Embedding:
        faces = self._analyzer.detect_faces(image)
        if not faces:
            raise FaceNotDetectedError()
        largest = max(faces, key=lambda face: face.area)
        return self._analyzer.embed_detections(image, [largest])[0].embedding

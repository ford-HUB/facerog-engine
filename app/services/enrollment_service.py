import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.core.exceptions import FaceNotDetectedError, MultipleFacesError
from app.ml.embedder import Embedding
from app.ml.face_analyzer import FaceAnalyzer
from app.models.face_record import DatasetPerson, FaceRecord
from app.repositories.face_repository import FaceRepository, slugify_person_id
from app.repositories.vector_index import InMemoryVectorIndex
from app.utils.image import decode_image, read_image_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentSummary:
    person_count: int
    template_count: int
    skipped_images: tuple[str, ...]


class EnrollmentService:
    """Builds the gallery of known people from the dataset folder and from uploads."""

    def __init__(
        self, analyzer: FaceAnalyzer, repository: FaceRepository, vector_index: InMemoryVectorIndex
    ) -> None:
        self._analyzer = analyzer
        self._repository = repository
        self._vector_index = vector_index

    def enroll_dataset(self) -> EnrollmentSummary:
        """Re-enroll everyone in the dataset folder, replacing the current gallery."""
        self._vector_index.clear()
        self._repository.clear_records()
        skipped: list[str] = []
        template_count = 0
        for person in self._repository.scan_dataset():
            embeddings, person_skipped = self._embed_person(person)
            skipped.extend(person_skipped)
            if not embeddings:
                logger.warning("No usable face for %s; not enrolled", person.person_id)
                continue
            self._index_person(person, embeddings, image_count=len(person.image_paths) - len(person_skipped))
            template_count += len(embeddings)
        summary = EnrollmentSummary(len(self._repository.list_records()), template_count, tuple(skipped))
        logger.info(
            "Enrolled %d people (%d templates, %d images skipped)",
            summary.person_count, summary.template_count, len(summary.skipped_images),
        )
        return summary

    def enroll_face(self, name: str, image_data: bytes, extension: str) -> FaceRecord:
        """Enroll one uploaded photo. Adds to an existing person when the name matches."""
        image = decode_image(image_data)
        embeddings = self._embed_single_face(image)
        person_id = slugify_person_id(name)
        self._repository.save_image(person_id, name, image_data, extension)
        self._vector_index.add(person_id, embeddings)
        existing = self._repository.find_record(person_id)
        person = self._repository.read_person(person_id)
        record = FaceRecord(
            person_id=person_id,
            name=person.name,
            photo_path=existing.photo_path if existing else person.photo_path,
            image_count=(existing.image_count if existing else 0) + 1,
            template_count=(existing.template_count if existing else 0) + len(embeddings),
            embedding_model=self._analyzer.embedding_model,
            enrolled_at=datetime.now(UTC),
        )
        self._repository.save_record(record)
        logger.info("Enrolled photo for %s (%d templates)", person_id, record.template_count)
        return record

    def delete_face(self, person_id: str) -> None:
        """Remove a person from the gallery and delete their dataset folder."""
        self._repository.remove_record(person_id)
        self._vector_index.remove(person_id)
        self._repository.delete_person_folder(person_id)
        logger.info("Deleted %s", person_id)

    def list_faces(self) -> list[FaceRecord]:
        return self._repository.list_records()

    def get_face(self, person_id: str) -> FaceRecord:
        return self._repository.get_record(person_id)

    def _embed_person(self, person: DatasetPerson) -> tuple[list[Embedding], list[str]]:
        embeddings: list[Embedding] = []
        skipped: list[str] = []
        for path in person.image_paths:
            image = read_image_file(str(path))
            faces = self._analyzer.detect_faces(image) if image is not None else []
            if not faces:
                skipped.append(_display_path(path))
                continue
            largest = max(faces, key=lambda face: face.area)
            embeddings.extend(self._analyzer.embed_enrollment_templates(image, largest))
        return embeddings, skipped

    def _embed_single_face(self, image: np.ndarray) -> list[Embedding]:
        faces = self._analyzer.detect_faces(image)
        if not faces:
            raise FaceNotDetectedError()
        if len(faces) > 1:
            raise MultipleFacesError(f"{len(faces)} faces detected; upload a photo with one face.")
        return self._analyzer.embed_enrollment_templates(image, faces[0])

    def _index_person(self, person: DatasetPerson, embeddings: list[Embedding], image_count: int) -> None:
        self._vector_index.add(person.person_id, embeddings)
        self._repository.save_record(
            FaceRecord(
                person_id=person.person_id,
                name=person.name,
                photo_path=person.photo_path,
                image_count=image_count,
                template_count=len(embeddings),
                embedding_model=self._analyzer.embedding_model,
                enrolled_at=datetime.now(UTC),
            )
        )


def _display_path(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"

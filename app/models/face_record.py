from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FaceRecord:
    """An enrolled person.

    photo_path is the registered picture shown when this person is recognized.
    template_count is how many embeddings (photos plus their mirrored versions) are indexed.
    """

    person_id: str
    name: str
    photo_path: Path
    image_count: int
    template_count: int
    embedding_model: str
    enrolled_at: datetime


@dataclass(frozen=True)
class DatasetPerson:
    """A person folder found in the dataset directory, before enrollment."""

    person_id: str
    name: str
    image_paths: tuple[Path, ...]
    photo_path: Path

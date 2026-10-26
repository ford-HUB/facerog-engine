"""Enrolled people: the dataset folder on disk plus their in-memory records.

Dataset layout (one folder per person):

    datasets/faces/
        juan_dela_cruz/
            info.json        optional: {"name": "Juan Dela Cruz"}
            profile.jpg      optional: picture returned on recognition
            01.jpg, 02.jpg   photos used for enrollment (more angles = better)
"""

import json
import logging
import re
import shutil
import threading
import uuid
from pathlib import Path

from app.core.exceptions import FaceNotFoundError
from app.models.face_record import DatasetPerson, FaceRecord

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
INFO_FILE_NAME = "info.json"
PROFILE_STEM = "profile"


def slugify_person_id(name: str) -> str:
    """'Juan Dela Cruz' -> 'juan_dela_cruz'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or f"person_{uuid.uuid4().hex[:8]}"


def name_from_folder(folder_name: str) -> str:
    """'juan_dela_cruz' -> 'Juan Dela Cruz'."""
    return " ".join(part.capitalize() for part in re.split(r"[_\-\s]+", folder_name) if part)


class FaceRepository:
    """Reads and writes the dataset folder and keeps enrolled `FaceRecord`s in memory."""

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir
        self._records: dict[str, FaceRecord] = {}
        self._lock = threading.Lock()

    def scan_dataset(self) -> list[DatasetPerson]:
        """List person folders that contain at least one image. Folders starting with '.' or '_' are skipped."""
        if not self._dataset_dir.is_dir():
            logger.warning("Dataset directory %s does not exist", self._dataset_dir)
            return []
        people = []
        for folder in sorted(self._dataset_dir.iterdir()):
            if folder.is_file() and folder.suffix.lower() in IMAGE_EXTENSIONS:
                logger.warning(
                    "Ignoring %s: put photos in a per-person folder, e.g. %s/<person_name>/%s",
                    folder.name, self._dataset_dir, folder.name,
                )
                continue
            if not folder.is_dir() or folder.name.startswith((".", "_")):
                continue
            person = self.read_person(folder.name)
            if person.image_paths:
                people.append(person)
        return people

    def read_person(self, person_id: str) -> DatasetPerson:
        folder = self._dataset_dir / person_id
        image_paths = tuple(
            sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        ) if folder.is_dir() else ()
        profile = next((p for p in image_paths if p.stem.lower() == PROFILE_STEM), None)
        photo_path = profile or (image_paths[0] if image_paths else folder / "missing")
        return DatasetPerson(
            person_id=person_id,
            name=self._read_name(folder) or name_from_folder(person_id),
            image_paths=image_paths,
            photo_path=photo_path,
        )

    def save_image(self, person_id: str, name: str, image_data: bytes, extension: str) -> Path:
        """Store an uploaded enrollment photo in the person's dataset folder."""
        folder = self._dataset_dir / person_id
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / INFO_FILE_NAME
        if not info_path.exists():
            info_path.write_text(json.dumps({"name": name}, indent=2), encoding="utf-8")
        image_path = folder / f"{uuid.uuid4().hex[:12]}{extension}"
        image_path.write_bytes(image_data)
        return image_path

    def delete_person_folder(self, person_id: str) -> None:
        folder = self._dataset_dir / person_id
        if folder.is_dir() and folder.resolve().parent == self._dataset_dir.resolve():
            shutil.rmtree(folder)

    def save_record(self, record: FaceRecord) -> None:
        with self._lock:
            self._records[record.person_id] = record

    def get_record(self, person_id: str) -> FaceRecord:
        with self._lock:
            record = self._records.get(person_id)
        if record is None:
            raise FaceNotFoundError(f"No enrolled person with id '{person_id}'.")
        return record

    def find_record(self, person_id: str) -> FaceRecord | None:
        with self._lock:
            return self._records.get(person_id)

    def list_records(self) -> list[FaceRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.name.lower())

    def remove_record(self, person_id: str) -> None:
        with self._lock:
            if self._records.pop(person_id, None) is None:
                raise FaceNotFoundError(f"No enrolled person with id '{person_id}'.")

    def clear_records(self) -> None:
        with self._lock:
            self._records.clear()

    @staticmethod
    def _read_name(folder: Path) -> str | None:
        info_path = folder / INFO_FILE_NAME
        if not info_path.is_file():
            return None
        try:
            name = json.loads(info_path.read_text(encoding="utf-8")).get("name")
        except (json.JSONDecodeError, AttributeError) as err:
            logger.warning("Ignoring invalid %s: %s", info_path, err)
            return None
        return name if isinstance(name, str) and name.strip() else None

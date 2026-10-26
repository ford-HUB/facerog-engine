import json
from pathlib import Path

from app.repositories.face_repository import FaceRepository, name_from_folder, slugify_person_id


def test_slugify_person_id_lowercases_and_joins_words() -> None:
    assert slugify_person_id("  Juan Dela Cruz ") == "juan_dela_cruz"


def test_name_from_folder_title_cases_words() -> None:
    assert name_from_folder("juan_dela_cruz") == "Juan Dela Cruz"


def test_scan_dataset_reads_info_name_and_profile_photo(tmp_path: Path) -> None:
    person_dir = tmp_path / "maria"
    person_dir.mkdir()
    (person_dir / "01.jpg").write_bytes(b"x")
    (person_dir / "profile.jpg").write_bytes(b"x")
    (person_dir / "info.json").write_text(json.dumps({"name": "María Santos"}), encoding="utf-8")
    (tmp_path / "_ignored").mkdir()
    (tmp_path / "_ignored" / "01.jpg").write_bytes(b"x")

    people = FaceRepository(tmp_path).scan_dataset()

    assert [p.person_id for p in people] == ["maria"]
    assert people[0].name == "María Santos"
    assert people[0].photo_path.name == "profile.jpg"
    assert len(people[0].image_paths) == 2


def test_scan_dataset_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert FaceRepository(tmp_path / "missing").scan_dataset() == []

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ServiceContainer, wire_services
from app.core.config import Settings
from app.main import create_app
from app.ml.face_analyzer import FaceAnalyzer
from tests.conftest import FACE_COLORS, FakeDetector, FakeEmbedder, encode_png, make_face_image


class TwoSidedDetector:
    """Treats the left and right halves of the image as two separate faces."""

    def detect(self, image: np.ndarray):
        width = image.shape[1]
        faces = []
        for offset in (0, width // 2):
            half = image[:, offset : offset + width // 2]
            for face in FakeDetector().detect(half):
                shift = np.array([offset, 0], dtype=np.float32)
                faces.append(
                    type(face)(
                        bbox=face.bbox + np.tile(shift, 2),
                        landmarks=face.landmarks + shift,
                        score=face.score,
                    )
                )
        return faces


def _open_tv(client: TestClient) -> None:
    """The TV's poll is its heartbeat; camera check-ins happen only while it polls."""
    client.get("/api/v1/check-ins")


def _two_people_frame() -> np.ndarray:
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[:, :400] = FACE_COLORS["red"]
    frame[:, 400:] = FACE_COLORS["blue"]
    return frame


@pytest.fixture
def two_face_container(settings: Settings) -> ServiceContainer:
    still = FaceAnalyzer(FakeDetector(), FakeEmbedder(), min_face_size_pixels=10)
    live = FaceAnalyzer(TwoSidedDetector(), FakeEmbedder(), min_face_size_pixels=10)
    return wire_services(settings, still, live)


@pytest.fixture
def client(settings: Settings, two_face_container: ServiceContainer) -> Iterator[TestClient]:
    app = create_app(settings, container_factory=lambda _: two_face_container)
    with TestClient(app) as test_client:
        yield test_client


def test_two_matched_faces_are_checked_in_together(
    client: TestClient, two_face_container: ServiceContainer
) -> None:
    client.post(
        "/api/v1/faces",
        data={"name": "Blue Person"},
        files={"image": ("blue.png", encode_png(make_face_image("blue")), "image/png")},
    )
    _open_tv(client)

    for _ in range(3):
        two_face_container.tracking_service.process_frame(_two_people_frame())

    body = client.get("/api/v1/check-ins").json()
    assert body["display_seconds"] == 3.0
    assert sorted(c["person"]["name"] for c in body["check_ins"]) == ["Blue Person", "Red Person"]
    assert all(c["status"] == "checked_in" for c in body["check_ins"])
    assert client.get(f"/api/v1/check-ins?after_sequence={body['last_sequence']}").json()["check_ins"] == []


def test_unknown_face_is_not_checked_in(client: TestClient, two_face_container: ServiceContainer) -> None:
    _open_tv(client)
    for _ in range(3):
        two_face_container.tracking_service.process_frame(_two_people_frame())

    names = [c["person"]["name"] for c in client.get("/api/v1/check-ins").json()["check_ins"]]
    assert names == ["Red Person"]


def test_reset_check_ins_clears_the_queue(client: TestClient, two_face_container: ServiceContainer) -> None:
    _open_tv(client)
    two_face_container.tracking_service.process_frame(_two_people_frame())
    session_before = client.get("/api/v1/check-ins").json()["session_id"]

    assert client.delete("/api/v1/check-ins").status_code == 204

    body = client.get("/api/v1/check-ins").json()
    assert body["check_ins"] == []
    assert body["session_id"] != session_before


def test_camera_checks_no_one_in_while_tv_mode_is_closed(
    client: TestClient, two_face_container: ServiceContainer
) -> None:
    tracking = two_face_container.tracking_service
    tracking.process_frame(_two_people_frame())

    assert tracking.get_snapshot().tracks[0].identification.person is not None
    assert two_face_container.check_in_service.list_check_ins() == []


def test_tv_page_is_served(client: TestClient) -> None:
    response = client.get("/tv")
    assert response.status_code == 200
    assert "Check-ins" in response.text


def test_removed_check_in_returns_while_person_stays_in_view(
    settings: Settings, client: TestClient, two_face_container: ServiceContainer
) -> None:
    tracking = two_face_container.tracking_service
    _open_tv(client)
    tracking.process_frame(_two_people_frame())
    first = client.get("/api/v1/check-ins").json()["check_ins"][0]

    assert client.delete(f"/api/v1/check-ins/{first['sequence']}").status_code == 204
    assert client.get("/api/v1/check-ins").json()["check_ins"] == []

    for _ in range(settings.reverify_interval_frames):
        tracking.process_frame(_two_people_frame())

    names = [c["person"]["name"] for c in client.get("/api/v1/check-ins").json()["check_ins"]]
    assert names == ["Red Person"]


def test_remove_unknown_check_in_returns_404(client: TestClient) -> None:
    assert client.delete("/api/v1/check-ins/99").status_code == 404


@pytest.fixture
def photo_client(settings: Settings, dataset_dir) -> Iterator[TestClient]:
    """Uploaded photos are split into two faces; Red and Blue are both registered."""
    blue_dir = dataset_dir / "blue_person"
    blue_dir.mkdir()
    (blue_dir / "01.png").write_bytes(encode_png(make_face_image("blue")))
    analyzer = FaceAnalyzer(TwoSidedDetector(), FakeEmbedder(), min_face_size_pixels=10)
    container = wire_services(settings, analyzer, analyzer)
    app = create_app(settings, container_factory=lambda _: container)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, image: np.ndarray):
    files = {"image": ("group.png", encode_png(image), "image/png")}
    return client.post("/api/v1/check-ins", files=files)


def test_uploaded_group_photo_queues_every_registered_face(photo_client: TestClient) -> None:
    response = _upload(photo_client, _two_people_frame())

    assert response.status_code == 200
    body = response.json()
    assert body["checked_in_count"] == 2
    assert [f["status"] for f in body["faces"]] == ["checked_in", "checked_in"]
    queue = photo_client.get("/api/v1/check-ins").json()["check_ins"]
    assert [c["person"]["name"] for c in queue] == ["Red Person", "Blue Person"]
    assert all(c["source"] == "photo" and c["track_id"] is None for c in queue)


def test_same_photo_uploaded_again_is_already_checked_in(photo_client: TestClient) -> None:
    _upload(photo_client, _two_people_frame())

    body = _upload(photo_client, _two_people_frame()).json()

    assert body["checked_in_count"] == 0
    assert [f["status"] for f in body["faces"]] == ["already_checked_in", "already_checked_in"]
    assert len(photo_client.get("/api/v1/check-ins").json()["check_ins"]) == 2


def test_uploaded_photo_checks_in_only_registered_faces(photo_client: TestClient) -> None:
    frame = _two_people_frame()
    frame[:, 400:] = FACE_COLORS["green"]

    body = _upload(photo_client, frame).json()

    assert [f["status"] for f in body["faces"]] == ["checked_in", "not_registered"]
    assert body["faces"][1]["person"] is None


def test_uploaded_photo_without_faces_returns_422(photo_client: TestClient) -> None:
    assert _upload(photo_client, np.zeros((100, 100, 3), dtype=np.uint8)).status_code == 422

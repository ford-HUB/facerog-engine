from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ServiceContainer
from app.core.config import Settings
from app.main import create_app
from tests.conftest import encode_png, make_face_image


@pytest.fixture
def client(settings: Settings, container: ServiceContainer) -> Iterator[TestClient]:
    app = create_app(settings, container_factory=lambda _: container)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_enrolled_people(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["enrolled_people"] == 1


def test_recognize_returns_name_and_photo_for_enrolled_face(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recognize", files={"image": ("face.png", encode_png(make_face_image("red")), "image/png")}
    )
    assert response.status_code == 200
    face = response.json()["faces"][0]
    assert face["is_match"] is True
    assert face["person"]["name"] == "Red Person"
    photo = client.get(face["person"]["photo_url"])
    assert photo.status_code == 200
    assert photo.headers["content-type"].startswith("image/")


def test_recognize_unknown_face_has_no_person(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recognize", files={"image": ("face.png", encode_png(make_face_image("blue")), "image/png")}
    )
    face = response.json()["faces"][0]
    assert face["is_match"] is False
    assert face["person"] is None


def test_recognize_rejects_non_image(client: TestClient) -> None:
    response = client.post("/api/v1/recognize", files={"image": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_enroll_then_list_faces(client: TestClient) -> None:
    response = client.post(
        "/api/v1/faces",
        data={"name": "Blue Person"},
        files={"image": ("blue.png", encode_png(make_face_image("blue")), "image/png")},
    )
    assert response.status_code == 201
    names = [face["name"] for face in client.get("/api/v1/faces").json()["faces"]]
    assert names == ["Blue Person", "Red Person"]


def test_delete_unknown_face_returns_404(client: TestClient) -> None:
    assert client.delete("/api/v1/faces/nobody").status_code == 404


def test_tracks_endpoint_when_stream_stopped(client: TestClient) -> None:
    body = client.get("/api/v1/stream/tracks").json()
    assert body["is_running"] is False
    assert body["tracks"] == []


def test_dashboard_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Faces in view" in response.text

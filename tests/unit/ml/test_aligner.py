import math

import numpy as np

from app.ml.aligner import ARCFACE_TEMPLATE, align_face, estimate_similarity_transform


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def test_estimate_similarity_transform_recovers_rotation_scale_translation() -> None:
    angle = math.radians(20)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    source = ARCFACE_TEMPLATE @ (2.5 * rotation).T + np.array([300.0, 120.0])
    matrix = estimate_similarity_transform(source.astype(np.float32), ARCFACE_TEMPLATE)
    np.testing.assert_allclose(_apply(matrix, source), ARCFACE_TEMPLATE, atol=1e-3)


def test_align_face_returns_112_square_crop() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = ARCFACE_TEMPLATE * 2 + 100
    assert align_face(image, landmarks).shape == (112, 112, 3)

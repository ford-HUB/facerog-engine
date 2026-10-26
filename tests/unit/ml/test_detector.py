import numpy as np

from app.ml.detector import distance_to_bbox, non_max_suppression, offsets_to_landmarks


def test_distance_to_bbox_expands_around_anchor() -> None:
    points = np.array([[50.0, 40.0]], dtype=np.float32)
    distances = np.array([[10.0, 5.0, 20.0, 15.0]], dtype=np.float32)
    np.testing.assert_allclose(distance_to_bbox(points, distances), [[40.0, 35.0, 70.0, 55.0]])


def test_offsets_to_landmarks_returns_five_points_per_anchor() -> None:
    points = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    offsets = np.ones((2, 10), dtype=np.float32)
    landmarks = offsets_to_landmarks(points, offsets)
    assert landmarks.shape == (2, 5, 2)
    np.testing.assert_allclose(landmarks[1, 0], [21.0, 21.0])


def test_non_max_suppression_drops_overlapping_lower_scores() -> None:
    bboxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert non_max_suppression(bboxes, scores, iou_threshold=0.4) == [0, 2]

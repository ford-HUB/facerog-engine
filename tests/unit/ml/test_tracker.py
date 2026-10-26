import numpy as np

from app.ml.tracker import ByteTracker, TrackerConfig, iou_distance


def _box(x: float, y: float, size: float = 80) -> list[float]:
    return [x, y, x + size, y + size]


def test_tracker_moving_face_keeps_same_track_id() -> None:
    tracker = ByteTracker(TrackerConfig())
    track_ids = set()
    for step in range(10):
        tracks = tracker.update(np.array([_box(100 + step * 5, 100)]), np.array([0.9]))
        track_ids.update(t.track_id for t in tracks)
    assert track_ids == {1}


def test_tracker_two_faces_get_distinct_ids() -> None:
    tracker = ByteTracker(TrackerConfig())
    bboxes = np.array([_box(50, 50), _box(400, 50)])
    for _ in range(3):
        tracks = tracker.update(bboxes, np.array([0.9, 0.9]))
    assert sorted(t.track_id for t in tracks) == [1, 2]


def test_tracker_low_score_detection_keeps_track_alive() -> None:
    tracker = ByteTracker(TrackerConfig())
    for _ in range(3):
        tracker.update(np.array([_box(100, 100)]), np.array([0.9]))
    tracks = tracker.update(np.array([_box(102, 100)]), np.array([0.3]))
    assert [t.track_id for t in tracks] == [1]
    assert tracks[0].detection_index == 0


def test_tracker_face_returning_within_buffer_is_reidentified() -> None:
    tracker = ByteTracker(TrackerConfig(buffer_frames=10))
    for _ in range(3):
        tracker.update(np.array([_box(100, 100)]), np.array([0.9]))
    for _ in range(5):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
    tracks = tracker.update(np.array([_box(100, 100)]), np.array([0.9]))
    assert [t.track_id for t in tracks] == [1]


def test_iou_distance_identical_boxes_is_zero() -> None:
    box = np.array(_box(0, 0))
    assert iou_distance([box], [box])[0, 0] == 0.0

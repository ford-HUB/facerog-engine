"""ByteTrack multi-object tracker (Zhang et al., 2022), adapted for face boxes.

ByteTrack associates high-confidence detections first, then uses low-confidence ones
to keep existing tracks alive through blur or partial occlusion. A stable track id lets
the engine recognize a face once and reuse the identity instead of re-running ArcFace
on every frame.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.ml.kalman_filter import KalmanFilter

NO_DETECTION = -1
DUPLICATE_IOU_DISTANCE = 0.15


class TrackState(Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(frozen=True)
class TrackerConfig:
    high_threshold: float = 0.6
    low_threshold: float = 0.2
    new_track_threshold: float = 0.7
    buffer_frames: int = 30
    first_match_distance: float = 0.8
    second_match_distance: float = 0.5
    unconfirmed_match_distance: float = 0.7


class Track:
    """One tracked face. `detection_index` points at this frame's detection, or -1."""

    def __init__(self, bbox: np.ndarray, score: float, detection_index: int) -> None:
        self.bbox = bbox.astype(np.float64)
        self.score = score
        self.detection_index = detection_index
        self.track_id = 0
        self.state = TrackState.TRACKED
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    def activate(self, kalman: KalmanFilter, track_id: int, frame_id: int) -> None:
        self.track_id = track_id
        self.mean, self.covariance = kalman.initiate(bbox_to_xyah(self.bbox))
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, kalman: KalmanFilter, detection: "Track", frame_id: int) -> None:
        assert self.mean is not None and self.covariance is not None
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance, bbox_to_xyah(detection.bbox)
        )
        self.bbox = xyah_to_bbox(self.mean[:4])
        self.score = detection.score
        self.detection_index = detection.detection_index
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id

    def predict(self, kalman: KalmanFilter) -> None:
        assert self.mean is not None and self.covariance is not None
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0
        self.mean, self.covariance = kalman.predict(mean, self.covariance)
        self.bbox = xyah_to_bbox(self.mean[:4])

    @property
    def age_frames(self) -> int:
        return self.frame_id - self.start_frame


class ByteTracker:
    """Tracks face boxes across frames. Call `update` once per frame."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._kalman = KalmanFilter()
        self._frame_id = 0
        self._next_id = 0
        self._tracked: list[Track] = []
        self._lost: list[Track] = []

    def update(self, bboxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """Advance one frame with (N, 4) x1y1x2y2 boxes and (N,) scores; return active tracks."""
        self._frame_id += 1
        for track in [*self._tracked, *self._lost]:
            track.detection_index = NO_DETECTION
        high, low = self._split_detections(bboxes, scores)

        confirmed = [t for t in self._tracked if t.is_activated]
        unconfirmed = [t for t in self._tracked if not t.is_activated]
        pool = _merge_tracks(confirmed, self._lost)
        for track in pool:
            track.predict(self._kalman)

        updated, pool_unmatched, high_unmatched = self._associate(
            pool, high, self._config.first_match_distance, fuse_scores=True
        )
        still_tracked = [pool[i] for i in pool_unmatched if pool[i].state == TrackState.TRACKED]
        updated_low, low_unmatched_tracks, _ = self._associate(
            still_tracked, low, self._config.second_match_distance, fuse_scores=False
        )
        updated += updated_low
        newly_lost = [still_tracked[i] for i in low_unmatched_tracks]
        for track in newly_lost:
            track.state = TrackState.LOST

        remaining_high = [high[i] for i in high_unmatched]
        updated_unconfirmed, unconfirmed_unmatched, new_detections = self._associate(
            unconfirmed, remaining_high, self._config.unconfirmed_match_distance, fuse_scores=True
        )
        updated += updated_unconfirmed
        removed = {id(unconfirmed[i]) for i in unconfirmed_unmatched}

        started = self._start_tracks([remaining_high[i] for i in new_detections])
        self._refresh_lists(updated + started, newly_lost, removed)
        return [t for t in self._tracked if t.is_activated]

    def _split_detections(self, bboxes: np.ndarray, scores: np.ndarray) -> tuple[list[Track], list[Track]]:
        high, low = [], []
        for index, (bbox, score) in enumerate(zip(bboxes, scores, strict=True)):
            if score >= self._config.high_threshold:
                high.append(Track(bbox, float(score), index))
            elif score > self._config.low_threshold:
                low.append(Track(bbox, float(score), index))
        return high, low

    def _associate(
        self, tracks: list[Track], detections: list[Track], max_distance: float, fuse_scores: bool
    ) -> tuple[list[Track], list[int], list[int]]:
        """Match tracks to detections by IoU; update matched tracks in place.

        Returns (updated tracks, unmatched track indices, unmatched detection indices).
        """
        distances = iou_distance([t.bbox for t in tracks], [d.bbox for d in detections])
        if fuse_scores and distances.size:
            scores = np.array([d.score for d in detections])
            distances = 1 - (1 - distances) * scores[None, :]
        matches, unmatched_tracks, unmatched_detections = linear_assignment(distances, max_distance)
        updated = []
        for track_index, detection_index in matches:
            track = tracks[track_index]
            track.update(self._kalman, detections[detection_index], self._frame_id)
            updated.append(track)
        return updated, unmatched_tracks, unmatched_detections

    def _start_tracks(self, detections: list[Track]) -> list[Track]:
        started = []
        for detection in detections:
            if detection.score < self._config.new_track_threshold:
                continue
            self._next_id += 1
            detection.activate(self._kalman, self._next_id, self._frame_id)
            started.append(detection)
        return started

    def _refresh_lists(self, updated: list[Track], newly_lost: list[Track], removed: set[int]) -> None:
        updated_ids = {id(t) for t in updated}
        tracked = [
            t for t in self._tracked
            if t.state == TrackState.TRACKED and id(t) not in removed and id(t) not in updated_ids
        ]
        tracked = _merge_tracks(tracked, updated)
        lost = [t for t in self._lost if id(t) not in updated_ids]
        lost = _merge_tracks(lost, newly_lost)
        expired = {
            id(t) for t in lost if self._frame_id - t.frame_id > self._config.buffer_frames
        }
        lost = [t for t in lost if id(t) not in expired]
        tracked = [t for t in tracked if t.state == TrackState.TRACKED]
        self._tracked, self._lost = _remove_duplicates(tracked, lost)


def bbox_to_xyah(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, max(y2 - y1, 1e-6)
    return np.array([x1 + width / 2, y1 + height / 2, width / height, height])


def xyah_to_bbox(xyah: np.ndarray) -> np.ndarray:
    center_x, center_y, aspect, height = xyah
    width = aspect * height
    return np.array([center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2])


def iou_distance(first: list[np.ndarray], second: list[np.ndarray]) -> np.ndarray:
    """1 - IoU between every pair of x1y1x2y2 boxes, shape (len(first), len(second))."""
    if not first or not second:
        return np.zeros((len(first), len(second)))
    a = np.asarray(first)[:, None, :]
    b = np.asarray(second)[None, :, :]
    overlap_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    overlap_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    intersection = overlap_w * overlap_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return 1 - intersection / np.maximum(area_a + area_b - intersection, 1e-6)


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching, dropping pairs whose cost exceeds max_cost."""
    rows, cols = cost.shape
    if cost.size == 0:
        return [], list(range(rows)), list(range(cols))
    row_indices, col_indices = linear_sum_assignment(cost)
    matches = [
        (int(r), int(c)) for r, c in zip(row_indices, col_indices, strict=True) if cost[r, c] <= max_cost
    ]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


def _merge_tracks(first: list[Track], second: list[Track]) -> list[Track]:
    seen = {t.track_id for t in first}
    return first + [t for t in second if t.track_id not in seen]


def _remove_duplicates(tracked: list[Track], lost: list[Track]) -> tuple[list[Track], list[Track]]:
    """When a tracked and a lost track overlap almost fully, keep only the older one."""
    distances = iou_distance([t.bbox for t in tracked], [t.bbox for t in lost])
    drop_tracked, drop_lost = set(), set()
    for tracked_index, lost_index in zip(*np.where(distances < DUPLICATE_IOU_DISTANCE), strict=True):
        if tracked[tracked_index].age_frames > lost[lost_index].age_frames:
            drop_lost.add(lost_index)
        else:
            drop_tracked.add(tracked_index)
    return (
        [t for i, t in enumerate(tracked) if i not in drop_tracked],
        [t for i, t in enumerate(lost) if i not in drop_lost],
    )

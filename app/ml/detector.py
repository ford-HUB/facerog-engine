from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

FloatArray = npt.NDArray[np.float32]

SCRFD_MEAN = 127.5
SCRFD_STD = 128.0

# Output layout of SCRFD exports, keyed by number of output tensors:
# (feature map count, strides, anchors per location, has landmarks)
_SCRFD_LAYOUTS: dict[int, tuple[int, tuple[int, ...], int, bool]] = {
    6: (3, (8, 16, 32), 2, False),
    9: (3, (8, 16, 32), 2, True),
    10: (5, (8, 16, 32, 64, 128), 1, False),
    15: (5, (8, 16, 32, 64, 128), 1, True),
}


@dataclass(frozen=True)
class DetectedFace:
    """A face found in an image.

    bbox is (x1, y1, x2, y2) in pixels; landmarks is (5, 2) in pixels, ordered
    left eye, right eye, nose tip, left mouth corner, right mouth corner.
    """

    bbox: FloatArray
    landmarks: FloatArray
    score: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[DetectedFace]: ...


class ScrfdDetector:
    """SCRFD face detector running on ONNX Runtime. Input images are BGR uint8."""

    def __init__(
        self,
        session: ort.InferenceSession,
        input_size: int,
        confidence_threshold: float,
        nms_threshold: float,
    ) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        layout = _SCRFD_LAYOUTS.get(len(self._output_names))
        if layout is None:
            raise ValueError(f"Unsupported SCRFD model with {len(self._output_names)} outputs")
        self._feature_map_count, self._strides, self._anchor_count, self._has_landmarks = layout
        if not self._has_landmarks:
            raise ValueError("SCRFD model without landmark outputs cannot be used for alignment")
        self._anchor_cache: dict[tuple[int, int], FloatArray] = {}

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces, returned sorted by confidence (highest first)."""
        blob, scale = self._preprocess(image)
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        scores, bboxes, landmarks = self._decode(outputs)
        if scores.size == 0:
            return []
        bboxes /= scale
        landmarks /= scale
        keep = non_max_suppression(bboxes, scores, self._nms_threshold)
        return [
            DetectedFace(bbox=bboxes[i], landmarks=landmarks[i], score=float(scores[i]))
            for i in keep
        ]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox into a square canvas (top-left anchored) and build an NCHW RGB blob."""
        height, width = image.shape[:2]
        scale = self._input_size / max(height, width)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        canvas = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / SCRFD_STD, canvas.shape[1::-1], (SCRFD_MEAN,) * 3, swapRB=True
        )
        return blob, scale

    def _decode(self, outputs: list[np.ndarray]) -> tuple[FloatArray, FloatArray, FloatArray]:
        all_scores, all_bboxes, all_landmarks = [], [], []
        count = self._feature_map_count
        for index, stride in enumerate(self._strides):
            scores = _squeeze_batch(outputs[index]).reshape(-1)
            bbox_distances = _squeeze_batch(outputs[index + count]) * stride
            landmark_offsets = _squeeze_batch(outputs[index + count * 2]) * stride
            anchors = self._anchor_centers(stride)
            positive = np.where(scores >= self._confidence_threshold)[0]
            all_scores.append(scores[positive])
            all_bboxes.append(distance_to_bbox(anchors[positive], bbox_distances[positive]))
            all_landmarks.append(
                offsets_to_landmarks(anchors[positive], landmark_offsets[positive])
            )
        scores = np.concatenate(all_scores).astype(np.float32)
        order = np.argsort(-scores)
        bboxes = np.concatenate(all_bboxes).astype(np.float32)[order]
        landmarks = np.concatenate(all_landmarks).astype(np.float32)[order]
        return scores[order], bboxes, landmarks

    def _anchor_centers(self, stride: int) -> FloatArray:
        cells = self._input_size // stride
        key = (cells, stride)
        if key not in self._anchor_cache:
            grid = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            centers = np.repeat(centers, self._anchor_count, axis=0)
            self._anchor_cache[key] = centers
        return self._anchor_cache[key]


def _squeeze_batch(output: np.ndarray) -> np.ndarray:
    return output[0] if output.ndim == 3 else output


def distance_to_bbox(points: FloatArray, distances: FloatArray) -> FloatArray:
    """Convert (left, top, right, bottom) distances from anchor points into x1y1x2y2 boxes."""
    return np.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        axis=-1,
    )


def offsets_to_landmarks(points: FloatArray, offsets: FloatArray) -> FloatArray:
    """Convert per-anchor (dx, dy) offsets for 5 landmarks into (N, 5, 2) pixel positions."""
    return offsets.reshape(-1, 5, 2) + points[:, None, :]


def non_max_suppression(bboxes: FloatArray, scores: FloatArray, iou_threshold: float) -> list[int]:
    """Greedy NMS. Returns indices of kept boxes, highest score first."""
    x1, y1, x2, y2 = bboxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        overlap_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]) + 1)
        overlap_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]) + 1)
        intersection = overlap_w * overlap_h
        iou = intersection / (areas[best] + areas[rest] - intersection)
        order = rest[iou <= iou_threshold]
    return keep

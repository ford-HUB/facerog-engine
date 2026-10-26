"""Rendering for the annotated video stream and the zoomed face-structure view."""

import cv2
import numpy as np

from app.utils.image import crop_square

MATCHED_COLOR = (80, 200, 60)
UNKNOWN_COLOR = (60, 60, 230)
PENDING_COLOR = (0, 190, 240)
TEXT_COLOR = (255, 255, 255)
STRUCTURE_COLOR = (255, 220, 120)
LANDMARK_COLOR = (0, 255, 255)

ZOOM_MARGIN_RATIO = 0.25
INSET_SIZE_PIXELS = 140
MAX_INSETS = 4

STRUCTURE_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (0, 3), (1, 4))


def draw_track_box(frame: np.ndarray, bbox: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(y1 - text_h - baseline - 6, 0)
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 6), color, -1)
    cv2.putText(frame, label, (x1 + 4, top + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


def render_zoom(
    frame: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray, size: int, caption: str
) -> np.ndarray:
    """Zoom into a face and overlay its landmark structure.

    The crop is upscaled with bicubic interpolation so small or angled faces are large
    enough to inspect, and the eyes-nose-mouth mesh shows how the face is oriented.
    """
    crop, origin = crop_square(frame, bbox, ZOOM_MARGIN_RATIO)
    scale = size / max(crop.shape[0], 1)
    zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    points = ((landmarks - origin) * scale).astype(np.int32)
    for start, end in STRUCTURE_EDGES:
        cv2.line(zoom, tuple(points[start]), tuple(points[end]), STRUCTURE_COLOR, 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(zoom, tuple(point), 3, LANDMARK_COLOR, -1, cv2.LINE_AA)
    _draw_caption(zoom, caption)
    return zoom


def draw_zoom_insets(frame: np.ndarray, zooms: list[np.ndarray]) -> None:
    """Paste picture-in-picture zooms down the right edge of frame, in place."""
    height, width = frame.shape[:2]
    inset = min(INSET_SIZE_PIXELS, height // MAX_INSETS - 8)
    if inset <= 16:
        return
    for slot, zoom in enumerate(zooms[:MAX_INSETS]):
        top = 8 + slot * (inset + 8)
        left = width - inset - 8
        frame[top : top + inset, left : left + inset] = cv2.resize(zoom, (inset, inset))
        cv2.rectangle(frame, (left, top), (left + inset, top + inset), PENDING_COLOR, 2)


def _draw_caption(image: np.ndarray, caption: str) -> None:
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, height - 24), (width, height), (0, 0, 0), -1)
    cv2.putText(image, caption, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

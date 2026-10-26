import cv2
import numpy as np

from app.core.exceptions import InvalidImageError

DEFAULT_JPEG_QUALITY = 85


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/WebP/BMP) into a BGR uint8 array."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError()
    return image


def read_image_file(path: str) -> np.ndarray | None:
    """Read an image from disk, tolerating non-ASCII paths (cv2.imread can't on Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def encode_jpeg(image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    is_encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not is_encoded:
        raise InvalidImageError("Image could not be encoded as JPEG.")
    return buffer.tobytes()


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Downscale so the image is at most `width` pixels wide, keeping aspect ratio."""
    height, current_width = image.shape[:2]
    if current_width <= width:
        return image
    scale = width / current_width
    return cv2.resize(image, (width, round(height * scale)), interpolation=cv2.INTER_AREA)


def crop_square(image: np.ndarray, bbox: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square region centered on bbox, enlarged by margin_ratio on each side.

    Returns (crop, origin) where origin is the crop's top-left (x, y) in image pixels.
    Regions outside the image are padded with black.
    """
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin_ratio)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(center_x - side / 2), round(center_y - side / 2)
    size = max(round(side), 1)
    height, width = image.shape[:2]
    pad = max(0, -left, -top, left + size - width, top + size - height)
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image
    crop = padded[top + pad : top + pad + size, left + pad : left + pad + size]
    return crop, np.array([left, top], dtype=np.float32)

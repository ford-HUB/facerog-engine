from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import InvalidImageError, PayloadTooLargeError

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def read_image_upload(upload: UploadFile, max_size_bytes: int) -> tuple[bytes, str]:
    """Read an uploaded image, enforcing type and size before it is decoded.

    Returns (raw bytes, file extension to store it under).
    """
    extension = ALLOWED_CONTENT_TYPES.get(upload.content_type or "")
    if extension is None:
        suffix = Path(upload.filename or "").suffix.lower()
        extension = suffix if suffix in ALLOWED_CONTENT_TYPES.values() else None
    if extension is None:
        raise InvalidImageError(f"Unsupported content type '{upload.content_type}'.")
    data = upload.file.read(max_size_bytes + 1)
    if len(data) > max_size_bytes:
        raise PayloadTooLargeError(f"Image exceeds {max_size_bytes} bytes.")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    return data, extension

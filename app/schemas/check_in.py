from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.recognition import BoundingBox, PersonMatch


class CheckInResponse(BaseModel):
    sequence: int
    status: Literal["checked_in"] = "checked_in"
    source: Literal["camera", "photo"]
    person: PersonMatch
    similarity: float
    track_id: int | None = Field(description="Live track identified; null for uploaded photos.")
    checked_in_at: float


class CheckInListResponse(BaseModel):
    """Check-ins after the requested sequence. Poll again with after_sequence=last_sequence."""

    session_id: str
    last_sequence: int
    display_seconds: float
    check_ins: list[CheckInResponse]


class PhotoFaceCheckInResponse(BaseModel):
    """One face of an uploaded photo. check_in is set only when status is checked_in."""

    bbox: BoundingBox
    status: Literal["checked_in", "already_checked_in", "not_registered"]
    similarity: float
    person: PersonMatch | None
    check_in: CheckInResponse | None


class PhotoCheckInResponse(BaseModel):
    """Every face found in the photo, left to right. New check-ins join the TV queue together."""

    checked_in_count: int
    faces: list[PhotoFaceCheckInResponse]

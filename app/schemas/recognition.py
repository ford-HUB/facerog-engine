from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel coordinates of the face box in the analyzed image."""

    x1: float
    y1: float
    x2: float
    y2: float


class HeadPoseEstimate(BaseModel):
    """Approximate head orientation in degrees."""

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    is_angled: bool


class PersonMatch(BaseModel):
    """The recognized person: their name and registered picture."""

    person_id: str
    name: str
    photo_url: str


class CandidateMatch(BaseModel):
    person_id: str
    name: str
    similarity: float


class RecognizedFace(BaseModel):
    bbox: BoundingBox
    detection_score: float
    head_pose: HeadPoseEstimate
    is_match: bool
    similarity: float = Field(description="Cosine similarity to the best candidate, in [-1, 1].")
    person: PersonMatch | None
    candidates: list[CandidateMatch]


class RecognitionResponse(BaseModel):
    match_threshold: float
    faces: list[RecognizedFace]


class VerificationResponse(BaseModel):
    similarity: float
    match_threshold: float
    is_match: bool

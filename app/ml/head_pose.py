import math
from dataclasses import dataclass

import numpy as np

# Where the nose tip sits between the eye line (0) and mouth line (1) on a frontal face,
# taken from the ArcFace landmark template.
FRONTAL_NOSE_RATIO = 0.494
YAW_DEGREES_PER_UNIT = 60.0
PITCH_DEGREES_PER_UNIT = 120.0


@dataclass(frozen=True)
class HeadPose:
    """Approximate head orientation in degrees, estimated from 5 face landmarks.

    yaw > 0: face turned toward image right; pitch > 0: face tilted down; roll > 0: clockwise.
    """

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float

    @property
    def frontal_score(self) -> float:
        """1.0 for a fully frontal face, falling toward 0 as the head turns away."""
        yaw = math.radians(min(abs(self.yaw_degrees), 90.0))
        pitch = math.radians(min(abs(self.pitch_degrees), 90.0))
        return max(0.05, math.cos(yaw) * math.cos(pitch))

    def is_angled(self, threshold_degrees: float) -> bool:
        return max(abs(self.yaw_degrees), abs(self.pitch_degrees)) >= threshold_degrees


def estimate_head_pose(landmarks: np.ndarray) -> HeadPose:
    """Estimate yaw/pitch/roll from (5, 2) landmarks without a 3D model.

    Roll comes from the eye line; yaw from how far the nose drifts off the eye midpoint;
    pitch from where the nose sits between the eyes and the mouth.
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks.astype(np.float64)
    eye_vector = right_eye - left_eye
    roll = math.atan2(eye_vector[1], eye_vector[0])
    upright = _rotate_points(np.stack([left_eye, right_eye, nose, left_mouth, right_mouth]), -roll)
    left_eye, right_eye, nose, left_mouth, right_mouth = upright
    eye_distance = max(np.linalg.norm(right_eye - left_eye), 1e-6)
    eye_mid = (left_eye + right_eye) / 2
    mouth_mid = (left_mouth + right_mouth) / 2
    yaw_ratio = (nose[0] - eye_mid[0]) / (eye_distance / 2)
    face_height = max(mouth_mid[1] - eye_mid[1], 1e-6)
    pitch_ratio = (nose[1] - eye_mid[1]) / face_height - FRONTAL_NOSE_RATIO
    return HeadPose(
        yaw_degrees=float(np.clip(yaw_ratio, -1.5, 1.5) * YAW_DEGREES_PER_UNIT),
        pitch_degrees=float(np.clip(pitch_ratio, -0.75, 0.75) * PITCH_DEGREES_PER_UNIT),
        roll_degrees=math.degrees(roll),
    )


def _rotate_points(points: np.ndarray, angle_radians: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle_radians), math.sin(angle_radians)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    center = points.mean(axis=0)
    return (points - center) @ rotation.T + center

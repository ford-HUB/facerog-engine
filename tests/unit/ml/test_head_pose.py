import numpy as np

from app.ml.aligner import ARCFACE_TEMPLATE
from app.ml.head_pose import estimate_head_pose


def test_estimate_head_pose_frontal_template_is_near_zero() -> None:
    pose = estimate_head_pose(ARCFACE_TEMPLATE)
    assert abs(pose.yaw_degrees) < 3
    assert abs(pose.pitch_degrees) < 3
    assert not pose.is_angled(15)


def test_estimate_head_pose_nose_shifted_sideways_is_angled() -> None:
    landmarks = ARCFACE_TEMPLATE.copy()
    landmarks[2, 0] += 10
    pose = estimate_head_pose(landmarks)
    assert pose.yaw_degrees > 15
    assert pose.is_angled(15)
    assert pose.frontal_score < estimate_head_pose(ARCFACE_TEMPLATE).frontal_score


def test_estimate_head_pose_tilted_head_reports_roll() -> None:
    angle = np.radians(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pose = estimate_head_pose(ARCFACE_TEMPLATE @ rotation.T)
    assert abs(pose.roll_degrees - 30) < 1
    assert abs(pose.yaw_degrees) < 3

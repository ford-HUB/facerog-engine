"""Photometric variants of an aligned face, used to get more out of a single enrollment photo.

An uploaded ID-style photo is usually bright and sharp, while live webcam faces are often
dim and low resolution. Enrolling these variants alongside the original narrows that gap.
"""

import cv2
import numpy as np

DARKEN_GAMMA = 1.8
LOW_RESOLUTION_PIXELS = 40


def darken(face: np.ndarray, gamma: float = DARKEN_GAMMA) -> np.ndarray:
    """Simulate a dim or backlit scene with a gamma curve (gamma > 1 darkens)."""
    lookup = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(face, lookup)


def simulate_low_resolution(face: np.ndarray, pixels: int = LOW_RESOLUTION_PIXELS) -> np.ndarray:
    """Simulate a small, soft face far from the camera by downscaling and upscaling again."""
    height, width = face.shape[:2]
    small = cv2.resize(face, (pixels, pixels), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

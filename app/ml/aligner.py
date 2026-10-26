import cv2
import numpy as np

ALIGNED_FACE_SIZE = 112

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping source points onto target.

    Returns a 2x3 affine matrix (rotation, uniform scale, translation).
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection[1, 1] = -1
    rotation = u @ reflection @ vt
    source_variance = source_centered.var(axis=0).sum()
    scale = np.trace(np.diag(singular_values) @ reflection) / source_variance
    translation = target_mean - scale * rotation @ source_mean
    return np.hstack([scale * rotation, translation[:, None]]).astype(np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a face so its 5 landmarks sit on the ArcFace template.

    Removes in-plane rotation (head tilt) and normalizes scale, which is what lets
    slightly angled faces still produce comparable embeddings. Returns a 112x112 BGR crop.
    """
    matrix = estimate_similarity_transform(landmarks.astype(np.float32), ARCFACE_TEMPLATE)
    return cv2.warpAffine(
        image, matrix, (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE), borderMode=cv2.BORDER_REPLICATE
    )

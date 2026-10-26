import numpy as np

POSITION_NOISE_WEIGHT = 1.0 / 20
VELOCITY_NOISE_WEIGHT = 1.0 / 160


class KalmanFilter:
    """Constant-velocity Kalman filter over boxes in (center x, center y, aspect, height) space.

    State is 8-dimensional: (cx, cy, a, h, vcx, vcy, va, vh).
    """

    def __init__(self) -> None:
        ndim = 4
        self._motion = np.eye(2 * ndim)
        self._motion[:ndim, ndim:] = np.eye(ndim)
        self._observation = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)]
        height = measurement[3]
        std = [
            2 * POSITION_NOISE_WEIGHT * height,
            2 * POSITION_NOISE_WEIGHT * height,
            1e-2,
            2 * POSITION_NOISE_WEIGHT * height,
            10 * VELOCITY_NOISE_WEIGHT * height,
            10 * VELOCITY_NOISE_WEIGHT * height,
            1e-5,
            10 * VELOCITY_NOISE_WEIGHT * height,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        std = [
            POSITION_NOISE_WEIGHT * height,
            POSITION_NOISE_WEIGHT * height,
            1e-2,
            POSITION_NOISE_WEIGHT * height,
            VELOCITY_NOISE_WEIGHT * height,
            VELOCITY_NOISE_WEIGHT * height,
            1e-5,
            VELOCITY_NOISE_WEIGHT * height,
        ]
        motion_covariance = np.diag(np.square(std))
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + motion_covariance
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_covariance = self._project(mean, covariance)
        kalman_gain = np.linalg.solve(
            projected_covariance, (covariance @ self._observation.T).T
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_covariance @ kalman_gain.T
        return new_mean, new_covariance

    def _project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        std = [
            POSITION_NOISE_WEIGHT * height,
            POSITION_NOISE_WEIGHT * height,
            1e-1,
            POSITION_NOISE_WEIGHT * height,
        ]
        innovation_covariance = np.diag(np.square(std))
        projected_mean = self._observation @ mean
        projected_covariance = (
            self._observation @ covariance @ self._observation.T + innovation_covariance
        )
        return projected_mean, projected_covariance

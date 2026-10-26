from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

Embedding = npt.NDArray[np.float32]

ARCFACE_MEAN = 127.5
ARCFACE_STD = 127.5


class FaceEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_faces(self, aligned_faces: list[np.ndarray]) -> list[Embedding]: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors (last axis) to unit length. The single place embeddings get normalized."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class ArcFaceEmbedder:
    """ArcFace embedding model on ONNX Runtime.

    Input: aligned 112x112 BGR uint8 crops (see `aligner.align_face`).
    Output: L2-normalized float32 embeddings, so cosine similarity is a dot product.
    """

    def __init__(self, session: ort.InferenceSession, model_name: str) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._model_name = model_name
        batch_dim = session.get_inputs()[0].shape[0]
        self._supports_batch = not isinstance(batch_dim, int) or batch_dim > 1

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_faces(self, aligned_faces: list[np.ndarray]) -> list[Embedding]:
        if not aligned_faces:
            return []
        blob = cv2.dnn.blobFromImages(
            aligned_faces, 1.0 / ARCFACE_STD, (112, 112), (ARCFACE_MEAN,) * 3, swapRB=True
        )
        if self._supports_batch:
            raw = self._session.run(None, {self._input_name: blob})[0]
        else:
            raw = np.concatenate(
                [self._session.run(None, {self._input_name: item[None]})[0] for item in blob]
            )
        return list(l2_normalize(raw))

import logging
import sys
import threading
import time
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 2.0


class FrameSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read_latest(self) -> tuple[int, np.ndarray | None]: ...

    @property
    def is_open(self) -> bool: ...


def parse_video_source(source: str) -> int | str:
    """"0" -> webcam index 0; anything else is a file path or stream URL (rtsp://, http://)."""
    return int(source) if source.isdigit() else source


def open_capture(source: int | str) -> cv2.VideoCapture:
    """Open a video source. Windows webcams use DirectShow first, because the default
    Media Foundation backend often fails with "can't grab frame" on laptop cameras."""
    if isinstance(source, int) and sys.platform == "win32":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(source)


class OpenCvVideoStream:
    """Reads frames on a background thread and keeps only the newest one.

    Dropping stale frames keeps latency low when inference is slower than the camera.
    Video files loop; cameras and streams reconnect after failures.
    """

    def __init__(self, source: str) -> None:
        self._source = parse_video_source(source)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_number = 0
        self._is_open = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="video-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def read_latest(self) -> tuple[int, np.ndarray | None]:
        """Return (frame number, frame). The number increases with every new frame."""
        with self._lock:
            return self._frame_number, self._frame

    def _run(self) -> None:
        while not self._stop_event.is_set():
            capture = open_capture(self._source)
            if not capture.isOpened():
                logger.warning("Cannot open video source %r; retrying", self._source)
                self._is_open = False
                self._stop_event.wait(RECONNECT_DELAY_SECONDS)
                continue
            self._is_open = True
            logger.info("Video source %r opened", self._source)
            self._read_until_failure(capture)
            capture.release()
        self._is_open = False

    def _read_until_failure(self, capture: cv2.VideoCapture) -> None:
        is_file = isinstance(self._source, str) and "://" not in self._source
        frame_interval = 1.0 / (capture.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
        while not self._stop_event.is_set():
            has_frame, frame = capture.read()
            if not has_frame:
                if is_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Video source %r stopped delivering frames", self._source)
                return
            with self._lock:
                self._frame = frame
                self._frame_number += 1
            if frame_interval:
                time.sleep(frame_interval)

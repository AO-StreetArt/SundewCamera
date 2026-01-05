"""Camera client that reads frames and pushes them onto a queue."""

from __future__ import annotations

import logging
import threading
from queue import Full, Queue
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)
import cv2


class _Capture(Protocol):
    def read(self):  # pragma: no cover - protocol stub
        ...

    def release(self) -> None:  # pragma: no cover - protocol stub
        ...


class CameraClient:
    """Reads frames from a camera device and enqueues them for processing."""

    def __init__(
        self,
        frame_queue: Queue,
        camera_index: int = 0,
        capture_factory: Optional[Callable[[int], _Capture]] = None,
    ) -> None:
        self._frame_queue = frame_queue
        self._camera_index = camera_index
        self._capture_factory = capture_factory
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-reader")
        self._thread.start()
        logger.info("Camera client started (index=%s)", self._camera_index)

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Camera client stopped")

    def _run(self) -> None:
        capture_factory = self._capture_factory or self._default_capture_factory
        cap = capture_factory(self._camera_index)
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read returned no frame")
                    break
                try:
                    self._frame_queue.put_nowait(frame)
                except Full:
                    # Drop frames if downstream is slower than capture.
                    logger.debug("Dropping frame because queue is full")
                    pass
        finally:
            cap.release()

    @staticmethod
    def _default_capture_factory(camera_index: int) -> _Capture:
        return cv2.VideoCapture(camera_index)

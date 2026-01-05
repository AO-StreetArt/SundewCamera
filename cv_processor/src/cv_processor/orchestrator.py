"""Processing orchestrator for CV pipeline."""

from __future__ import annotations

import json
import logging
from queue import Empty, Queue
from time import time
from typing import Any, Optional

import cv2

from sundew_common.ipc_client import ZmqIpcClient

from .camera_client import CameraClient
from .hailo_infer_client import HailoInferClient

logger = logging.getLogger(__name__)

class CvOrchestrator:
    """Builds and runs the CV pipeline."""

    def __init__(
        self,
        *,
        network: str,
        batch_size: int,
        ipc_endpoint: Optional[str],
        ipc_socket_type: str = "PUB",
        camera_index: int = 0,
        queue_maxsize: int = 4,
        output_console: bool = False,
        frame_stride: int = 1,
        resize_width: int = 640,
        resize_height: int = 640,
        frame_queue: Optional[Queue] = None,
        camera_client: Optional[CameraClient] = None,
        infer_client: Optional[HailoInferClient] = None,
        ipc_client: Optional[ZmqIpcClient] = None,
    ) -> None:
        self._frame_queue = frame_queue or Queue(maxsize=queue_maxsize)
        self._camera_client = camera_client or CameraClient(
            self._frame_queue, camera_index=camera_index
        )
        self._infer_client = infer_client or HailoInferClient(
            network, batch_size
        )
        self._output_console = output_console
        self._ipc_client = None
        if not self._output_console:
            if ipc_endpoint is None:
                raise ValueError("ipc_endpoint is required when not outputting to console")
            self._ipc_client = ipc_client or ZmqIpcClient(
                ipc_endpoint, ipc_socket_type
            )
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self._frame_stride = frame_stride
        if resize_width < 1 or resize_height < 1:
            raise ValueError("resize_width/resize_height must be >= 1")
        self._resize_width = resize_width
        self._resize_height = resize_height
        self._frame_id = 0

    def run(self, *, max_frames: Optional[int] = None) -> None:
        """Start the camera and run inference, sending results to IPC."""
        processed = 0
        self._camera_client.start()
        logger.info("CV orchestrator run loop started")
        try:
            while True:
                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except Empty:
                    continue
                if self._frame_id % self._frame_stride != 0:
                    self._frame_id += 1
                    continue
                callback = self._build_callback(self._frame_id)
                # TO-DO: call inference on a different thread to improve throughput.
                resized = self._resize_frame(frame)
                self._infer_client.run([resized], callback)
                self._frame_id += 1
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    logger.info("Reached max_frames=%s, stopping", max_frames)
                    break
        finally:
            self._camera_client.stop()
            self._infer_client.close()
            if self._ipc_client is not None:
                self._ipc_client.close()
            logger.info("CV orchestrator shut down")

    def _build_callback(self, frame_id: int):
        def _callback(completion_info: Any, bindings_list: Any) -> None:
            message = {
                "schema_version": "1.0",
                "timestamp_ms": int(time() * 1000),
                "frame_id": frame_id,
                "source": "camera-0",
                "model": "hailo-object-detect-v1",
                "detections": self._serialize_detections(bindings_list),
                "processing": {
                    "frame_stride": self._frame_stride,
                    "inference_ms": None,
                    "postprocess_ms": None,
                },
            }
            self._emit_message(message)

        return _callback

    def _serialize_detections(self, bindings_list: Any) -> list[Any]:
        # Minimal parsing: pass through raw bindings as detections.
        # TO-DO: implement proper parsing based on model output format.
        if bindings_list is None:
            return []
        if isinstance(bindings_list, list):
            return bindings_list
        return [bindings_list]

    def _emit_message(self, message: dict[str, Any]) -> None:
        if self._output_console:
            print(json.dumps(message), flush=True)
            return
        if self._ipc_client is None:
            raise RuntimeError("IPC client unavailable for message emit")
        self._ipc_client.send_json(message)

    def _resize_frame(self, frame: Any) -> Any:
        if frame is None or not hasattr(frame, "shape"):
            return frame
        return cv2.resize(frame, (self._resize_width, self._resize_height))

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return repr(value)

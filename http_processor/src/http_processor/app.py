"""Flask HTTP server for on-demand camera inference."""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, time
from typing import Any, Callable, Iterable, Optional

import cv2
from flask import Flask, jsonify, request

from cv_processor.detection_serializer import serialize_detections
from cv_processor.hailo_infer_client import HailoInferClient

logger = logging.getLogger(__name__)


class CameraCaptureError(RuntimeError):
    """Raised when the attached camera cannot provide a frame."""


class InferenceTimeoutError(TimeoutError):
    """Raised when Hailo does not complete within the request timeout."""


@dataclass(frozen=True)
class DetectionResult:
    detections: list[dict[str, Any]]
    capture_ms: float
    inference_ms: float


class CameraInferenceService:
    """Own a camera and Hailo client for synchronous request handling."""

    def __init__(
        self,
        network: str,
        *,
        camera_index: int = 0,
        timeout_seconds: float = 10.0,
        capture_factory: Optional[Callable[[int], Any]] = None,
        infer_client: Optional[HailoInferClient] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        factory = capture_factory or cv2.VideoCapture
        self._capture = factory(camera_index)
        self._infer_client = infer_client or HailoInferClient(network, batch_size=1)
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._closed = False

        is_opened = getattr(self._capture, "isOpened", None)
        if callable(is_opened) and not is_opened():
            self.close()
            raise CameraCaptureError(f"Unable to open camera index {camera_index}")

        input_shape = self._infer_client.get_input_shape()
        if len(input_shape) < 2:
            self.close()
            raise ValueError(f"Unsupported model input shape: {input_shape!r}")
        self._input_height = int(input_shape[0])
        self._input_width = int(input_shape[1])

    def detect(self) -> DetectionResult:
        """Capture one frame and wait for its asynchronous Hailo inference."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Camera inference service is closed")

            capture_started = monotonic()
            ok, frame = self._capture.read()
            capture_ms = (monotonic() - capture_started) * 1000
            if not ok or frame is None:
                raise CameraCaptureError("Camera did not return a frame")

            resized = cv2.resize(frame, (self._input_width, self._input_height))
            completed = threading.Event()
            callback_result: dict[str, Any] = {}

            def callback(completion_info: Any, bindings_list: Any) -> None:
                callback_result["completion_info"] = completion_info
                callback_result["bindings_list"] = bindings_list
                completed.set()

            inference_started = monotonic()
            self._infer_client.run([resized], callback)
            if not completed.wait(self._timeout_seconds):
                raise InferenceTimeoutError(
                    f"Hailo inference exceeded {self._timeout_seconds:g} seconds"
                )
            inference_ms = (monotonic() - inference_started) * 1000

            completion_info = callback_result.get("completion_info")
            exception = getattr(completion_info, "exception", None)
            if exception is not None:
                raise RuntimeError(f"Hailo inference failed: {exception}")

            return DetectionResult(
                detections=serialize_detections(callback_result.get("bindings_list")),
                capture_ms=capture_ms,
                inference_ms=inference_ms,
            )

    def close(self) -> None:
        """Release camera and Hailo resources. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        release = getattr(self._capture, "release", None)
        if callable(release):
            release()
        self._infer_client.close()


def create_app(
    service: CameraInferenceService,
    *,
    model_name: str,
    node_name: str = "pi-vision-01",
    service_name: str = "vision_sundew",
) -> Flask:
    """Create the one-endpoint Flask application."""
    app = Flask(__name__, static_folder=None)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "node": node_name,
                "service": service_name,
                "backend": {"name": "sundew-hailo", "status": "ok"},
                "capabilities": {
                    "detect_objects": {
                        "status": "ok",
                        "default_model": model_name,
                        "max_concurrency": 1,
                    }
                },
            }
        )

    @app.post("/detect")
    def detect():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            return jsonify({"error": "options must be a JSON object"}), 400
        confidence_threshold = options.get("confidence_threshold", 0.0)
        max_detections = options.get("max_detections")
        if (
            not isinstance(confidence_threshold, (int, float))
            or isinstance(confidence_threshold, bool)
            or not 0 <= confidence_threshold <= 1
        ):
            return jsonify({"error": "confidence_threshold must be between 0 and 1"}), 400
        if max_detections is not None and (
            not isinstance(max_detections, int) or isinstance(max_detections, bool) or max_detections < 1
        ):
            return jsonify({"error": "max_detections must be a positive integer"}), 400

        try:
            result = service.detect()
        except CameraCaptureError as exc:
            return jsonify({"error": str(exc)}), 503
        except InferenceTimeoutError as exc:
            return jsonify({"error": str(exc)}), 504
        except Exception as exc:  # pragma: no cover - exercised by hardware errors
            logger.exception("Detection request failed")
            return jsonify({"error": str(exc)}), 500

        detections = [
            item
            for item in result.detections
            if float(item.get("confidence", 1.0)) >= confidence_threshold
        ]
        if max_detections is not None:
            detections = detections[:max_detections]

        return jsonify(
            {
                "schema_version": "1.0",
                "timestamp_ms": int(time() * 1000),
                "source": "camera-0",
                "model": model_name,
                "detections": detections,
                "processing": {
                    "capture_ms": result.capture_ms,
                    "inference_ms": result.inference_ms,
                },
            }
        )

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Sundew HTTP processor.")
    parser.add_argument("--network", required=True, help="Path to the Hailo HEF")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--model-name", help="Stable model name returned by the API")
    parser.add_argument("--node-name", default="pi-vision-01")
    parser.add_argument("--service-name", default="vision_sundew")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    service = CameraInferenceService(
        args.network,
        camera_index=args.camera_index,
        timeout_seconds=args.timeout_seconds,
    )
    app = create_app(
        service,
        model_name=args.model_name or Path(args.network).stem,
        node_name=args.node_name,
        service_name=args.service_name,
    )
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

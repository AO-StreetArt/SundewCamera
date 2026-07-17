from unittest.mock import MagicMock

import numpy as np

from http_processor.app import (
    CameraCaptureError,
    CameraInferenceService,
    DetectionResult,
    create_app,
)


class FakeCapture:
    def __init__(self, frame=None, ok=True):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame
        self.ok = ok
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return self.ok, self.frame

    def release(self):
        self.released = True


def test_detect_captures_resizes_waits_and_returns_detections():
    capture = FakeCapture()
    infer = MagicMock()
    infer.get_input_shape.return_value = (320, 320, 3)

    def run(batch, callback):
        assert batch[0].shape == (320, 320, 3)
        callback(MagicMock(exception=None), [{"class_id": 0, "label": "person"}])

    infer.run.side_effect = run
    service = CameraInferenceService(
        "model.hef",
        capture_factory=lambda _index: capture,
        infer_client=infer,
    )

    result = service.detect()
    service.close()

    assert result.detections == [{"class_id": 0, "label": "person"}]
    assert capture.released is True
    infer.close.assert_called_once()


def test_endpoint_returns_formatted_result():
    service = MagicMock()
    service.detect.return_value = DetectionResult(
        detections=[{"label": "person", "confidence": 0.9}],
        capture_ms=2.0,
        inference_ms=4.0,
    )
    app = create_app(service, model_name="yolov6n_h8.hef")

    response = app.test_client().post("/detect")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model"] == "yolov6n_h8.hef"
    assert payload["detections"] == [{"confidence": 0.9, "label": "person"}]
    assert payload["processing"] == {"capture_ms": 2.0, "inference_ms": 4.0}


def test_health_describes_controller_capability():
    app = create_app(
        MagicMock(),
        model_name="sundew-detector",
        node_name="pi-vision-01",
        service_name="vision_sundew",
    )

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "node": "pi-vision-01",
        "service": "vision_sundew",
        "backend": {"name": "sundew-hailo", "status": "ok"},
        "capabilities": {
            "detect_objects": {
                "status": "ok",
                "default_model": "sundew-detector",
                "max_concurrency": 1,
            }
        },
    }


def test_detect_filters_by_confidence_and_count():
    service = MagicMock()
    service.detect.return_value = DetectionResult(
        detections=[
            {"label": "person", "confidence": 0.9},
            {"label": "cat", "confidence": 0.7},
            {"label": "chair", "confidence": 0.2},
        ],
        capture_ms=2.0,
        inference_ms=4.0,
    )
    app = create_app(service, model_name="sundew-detector")

    response = app.test_client().post(
        "/detect",
        json={"options": {"confidence_threshold": 0.5, "max_detections": 1}},
    )

    assert response.status_code == 200
    assert response.get_json()["detections"] == [{"label": "person", "confidence": 0.9}]


def test_detect_rejects_invalid_options_without_running_inference():
    service = MagicMock()
    app = create_app(service, model_name="sundew-detector")

    response = app.test_client().post(
        "/detect", json={"options": {"confidence_threshold": 2}}
    )

    assert response.status_code == 400
    service.detect.assert_not_called()


def test_endpoint_returns_503_when_camera_capture_fails():
    service = MagicMock()
    service.detect.side_effect = CameraCaptureError("Camera did not return a frame")
    app = create_app(service, model_name="model.hef")

    response = app.test_client().post("/detect")

    assert response.status_code == 503
    assert response.get_json() == {"error": "Camera did not return a frame"}


def test_get_is_not_supported():
    app = create_app(MagicMock(), model_name="model.hef")
    assert app.test_client().get("/detect").status_code == 405

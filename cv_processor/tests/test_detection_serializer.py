import numpy as np
import pytest

from cv_processor.detection_serializer import json_safe, safe_str, serialize_detections


class BadStr:
    def __str__(self) -> str:
        raise ValueError("boom")

    def __repr__(self) -> str:
        return "BadStr()"


class Key:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class FakeOutput:
    def __init__(self, buffer):
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer


class FakeBindings:
    def __init__(self, outputs):
        self.outputs = outputs
        self._output_names = list(outputs)

    def output(self, name):
        return FakeOutput(self.outputs[name])


def test_safe_str_falls_back_to_repr():
    assert safe_str(BadStr()) == "BadStr()"


def test_json_safe_handles_nested_containers():
    payload = {
        Key("k"): {"inner": (1, 2, 3), "set": {4, 5}},
        "value": BadStr(),
    }

    result = json_safe(payload)

    assert result["k"]["inner"] == [1, 2, 3]
    assert sorted(result["k"]["set"]) == [4, 5]
    assert result["value"] == "BadStr()"


def test_serialize_detections_wraps_single_value():
    assert serialize_detections(None) == []
    assert serialize_detections("det") == [{"value": "det"}]


def test_serialize_detections_normalizes_list_items():
    detections = [BadStr(), {"score": 0.9}]
    assert serialize_detections(detections) == [
        {"value": "BadStr()"},
        {"score": 0.9},
    ]


def test_serialize_detections_extracts_and_decodes_hailo_nms_buffer():
    # Hailo's TensorFlow-style NMS layout is classes x values x proposals.
    nms = np.zeros((2, 5, 3), dtype=np.float32)
    nms[0, :, 0] = [0.1, 0.2, 0.8, 0.7, 0.95]
    nms[1, :, 0] = [0.3, 0.4, 0.6, 0.9, 0.75]
    bindings = FakeBindings({"yolov6n_nms_postprocess": nms})

    result = serialize_detections([bindings])

    assert len(result) == 2
    assert result[0]["class_id"] == 0
    assert result[0]["label"] == "person"
    assert result[0]["confidence"] == pytest.approx(0.95)
    assert result[0]["bbox"] == pytest.approx(
        {"x_min": 0.2, "y_min": 0.1, "x_max": 0.7, "y_max": 0.8}
    )
    assert result[1]["class_id"] == 1
    assert result[1]["label"] == "bicycle"


def test_serialize_detections_supports_detection_major_nms_layout():
    nms = np.array([[[0.1, 0.2, 0.8, 0.7, 0.9]]], dtype=np.float32)
    bindings = FakeBindings({"nms": nms})

    result = serialize_detections([bindings])

    assert len(result) == 1
    assert result[0]["label"] == "person"


def test_serialize_detections_describes_unsupported_output_without_dumping_it():
    raw_head = np.zeros((80, 80, 24), dtype=np.float32)
    bindings = FakeBindings({"raw_yolo_head": raw_head})

    result = serialize_detections([bindings])

    assert result == [
        {
            "error": "unsupported model output; expected Hailo NMS tensor",
            "output_name": "raw_yolo_head",
            "shape": [80, 80, 24],
        }
    ]


def test_json_safe_handles_numpy_values():
    assert json_safe(np.float32(0.5)) == 0.5
    assert json_safe(np.array([1, 2])) == [1, 2]

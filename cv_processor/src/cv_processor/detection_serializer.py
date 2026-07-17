"""Extract and serialize object detections from Hailo inference bindings."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def serialize_detections(bindings_list: Any) -> list[dict[str, Any]]:
    """Extract Hailo output buffers and decode class-grouped NMS results.

    Hailo's async API gives the callback a list of ``Bindings`` objects, one
    for each input frame. A YOLO HEF with Hailo NMS normally exposes one tensor
    shaped ``(classes, 5, max_detections)``. The five values are
    ``y_min, x_min, y_max, x_max, score``.

    Plain Python values are still accepted to keep injected test clients and
    non-Hailo callers backwards compatible.
    """
    if bindings_list is None:
        return []

    bindings = bindings_list if isinstance(bindings_list, list) else [bindings_list]
    detections: list[dict[str, Any]] = []
    for binding in bindings:
        outputs = _extract_outputs(binding)
        if outputs is None:
            value = json_safe(binding)
            if isinstance(value, dict):
                detections.append(value)
            else:
                detections.append({"value": value})
            continue

        for output_name, output_buffer in outputs:
            decoded = _decode_nms_buffer(output_buffer)
            if decoded is not None:
                detections.extend(decoded)
            else:
                array = np.asarray(output_buffer)
                detections.append(
                    {
                        "error": "unsupported model output; expected Hailo NMS tensor",
                        "output_name": output_name,
                        "shape": list(array.shape),
                    }
                )
    return detections


def _extract_outputs(binding: Any) -> list[tuple[str, Any]] | None:
    """Return named buffers from a Hailo ``Bindings`` object, if applicable."""
    output = getattr(binding, "output", None)
    if not callable(output):
        return None

    names = getattr(binding, "_output_names", None)
    if names:
        return [(str(name), output(name).get_buffer()) for name in names]

    # Hailo allows output() without a name when the HEF has one output.
    try:
        return [("output", output().get_buffer())]
    except (TypeError, RuntimeError, ValueError):
        return None


def _decode_nms_buffer(buffer: Any) -> list[dict[str, Any]] | None:
    """Decode supported Hailo NMS-by-class representations."""
    class_boxes = _class_box_arrays(buffer)
    if class_boxes is None:
        return None

    detections: list[dict[str, Any]] = []
    for class_id, boxes in enumerate(class_boxes):
        for box in boxes:
            y_min, x_min, y_max, x_max, score = (float(value) for value in box)
            if not math.isfinite(score) or score <= 0.0:
                continue
            detection: dict[str, Any] = {
                "class_id": class_id,
                "confidence": score,
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
            }
            if class_id < len(COCO_LABELS):
                detection["label"] = COCO_LABELS[class_id]
            detections.append(detection)
    return detections


def _class_box_arrays(buffer: Any) -> Iterable[np.ndarray] | None:
    """Normalize NMS output to one ``(detections, 5)`` array per class."""
    if isinstance(buffer, (list, tuple)):
        arrays = [np.asarray(item) for item in buffer]
        if all(array.ndim == 2 and array.shape[-1] == 5 for array in arrays):
            return arrays
        return None

    array = np.asarray(buffer)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        return None
    if array.shape[-1] == 5:
        return array
    if array.shape[1] == 5:
        return array.transpose(0, 2, 1)
    return None


def safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)


def json_safe(value: Any) -> Any:
    """Convert common Python and NumPy values to JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {safe_str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return safe_str(value)

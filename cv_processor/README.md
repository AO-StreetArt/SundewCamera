# CV Processor

Python application that performs CV processing and publishes results via IPC.

Object-detection HEFs must include Hailo NMS post-processing. The processor
extracts each callback binding's output buffer and emits COCO class IDs,
labels, confidence scores, and normalized bounding boxes. Raw YOLO detection
heads are not decoded because their layout is specific to the compiled model.

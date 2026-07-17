# Sundew HTTP Processor

Flask service that captures one camera frame, runs it through a Hailo object
detection HEF, and returns JSON detections.

Install it after the CV processor:

```bash
pip install -e ./cv_processor
pip install -e ./http_processor
```

Run it on a Hailo-8 Raspberry Pi:

```bash
sundew-http-server \
  --network /usr/share/hailo-models/yolov6n_h8.hef \
  --model-name sundew-detector \
  --node-name pi-vision-01 \
  --host 0.0.0.0 \
  --port 8080
```

Controller-compatible health metadata is available at:

```bash
curl http://127.0.0.1:8080/health
```

Capture a frame and run inference:

```bash
curl -X POST http://127.0.0.1:8080/detect
```

The aiAgent controller can also pass camera detection options:

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{"options":{"confidence_threshold":0.35,"max_detections":100}}' \
  http://127.0.0.1:8080/detect
```

This endpoint currently captures from `camera-0`. Uploaded images and artifact
references are intentionally not supported yet.

The service keeps the camera and Hailo model open between calls. Requests are
processed one at a time because the camera and inference bindings are shared.

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
  --host 0.0.0.0 \
  --port 8080
```

Capture a frame and run inference:

```bash
curl -X POST http://127.0.0.1:8080/detect
```

The service keeps the camera and Hailo model open between calls. Requests are
processed one at a time because the camera and inference bindings are shared.

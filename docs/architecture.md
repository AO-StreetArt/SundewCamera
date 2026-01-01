# Architecture

Sundew prototypes are built on a Raspberry Pi 5 with an AI Hat, containing a Hailo AI processor, although if they are successful we can move onto a SOM from Hailo.ai with minimal work.

As far as software architecture - we need to optimize throughput well in order to keep up with the camera.

## Onboard Applications

We're going to have two high-level applications that run on the Raspberry Pi - a CV Processor, which performs ML tasks and then broadcasts the results; and an Output Processor, which takes the results and turns them into human-digestible form.  These are both python applications for our prototype, and will communicate via local IPC.

## CV Processor

Our CV Processor will be based on the Hailo-apps examples (ie. https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/object_detection.py).  We'll be following the same core process - a thread dedicated to CV processing on the Hailo processor, and then threads for pre/post processing.  In our case, the post processing thread will be setup to send its detected objects to a configured IPC address.  To begin with, we will just include object detection.

This will incorporate some basic optimizations - like only processing 1 in every X number of frames based on configuration.

## Output Processor

The Output Processor accepts IPC messages from the CV Processor, and then turns those into audio messages that can be played to the end user.  Right now, this is non-LLM based processing (so sound files stored on disk and then played to users based on a lookup table).  In the future, we'll evaluate if we can leverage a micro LLM on the Hailo board without compromising the performance of the CV features.

The output processor should receive an IPC message, and put it into a queue.  Then it can be taken off by another thread and processed.

## Proposed IPC Protocol

For the prototype, use ZeroMQ over a local IPC socket to keep latency low and avoid network overhead.  The CV Processor publishes messages; the Output Processor subscribes.  If you need guaranteed delivery later, you can swap to PUSH/PULL and add acks or local persistence.

- Transport: ZeroMQ PUB/SUB over `ipc:///tmp/sundew-cv.ipc`
- Direction: CV Processor -> Output Processor
- Wire format: UTF-8 JSON (compact), one message per detection batch
- Backpressure: Output Processor drops old messages if its queue is full to keep audio timely

## Proposed Message Schema

This is the minimal schema for object detection results.  It is versioned so both apps can evolve independently.

```json
{
  "schema_version": "1.0",
  "timestamp_ms": 1721586123456,
  "frame_id": 123456,
  "source": "camera-0",
  "model": "hailo-object-detect-v1",
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": { "x": 0.42, "y": 0.18, "w": 0.11, "h": 0.33 },
      "track_id": 17
    }
  ],
  "processing": {
    "frame_stride": 3,
    "inference_ms": 12.4,
    "postprocess_ms": 3.1
  }
}
```

Notes:
- `bbox` is normalized to `[0.0, 1.0]` with `x,y` as top-left corner.
- `frame_stride` records how many frames are skipped between processed frames.
- Optional fields can be omitted to keep messages small; unknown fields should be ignored.

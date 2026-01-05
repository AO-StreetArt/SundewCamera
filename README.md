# Sundew Camera

Sundew is a prototype IoT device that performs onboard computer vision tasks and provides flexible input and output channels.  User output is handled via headphones in order to support vision-impaired users.  The prototype device will run on a Raspberry Pi 5 with an AI Hat.

The overall purpose of the project is to evaluate the usage of new AI-enabled IoT chips and SoC/SoM vendors (specifically, in this case, Hailo.ai).

## Install + Run

The prototype is two Python services that communicate over local ZeroMQ IPC. The output processor prints detections as a placeholder for audio output.

### Prerequisites

- Raspberry Pi 5 with the AI Hat and Hailo runtime/SDK installed.
- Python 3.10+.
- A Hailo `.hef` model file. Follow the Hailo apps installation guide to obtain model files:
  `https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/installation.md`

### Install

From the repo root:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -U pip

pip install -e ./sundew_common
pip install -e ./output_processor
pip install -e ./cv_processor
```

### Run

Pick an IPC endpoint and run the output processor first (it binds by default):

```bash
output-processor --ipc-endpoint ipc:///tmp/sundew-cv.ipc
```

Then run the CV processor with your `.hef` model:

```bash
cv-processor --network /path/to/model.hef --ipc-endpoint ipc:///tmp/sundew-cv.ipc
```

### Configuration options

CV processor (`cv-processor`):
- `--network` (required): path to the Hailo `.hef` model.
- `--output-console`: print detections instead of sending via IPC.
- `--ipc-endpoint`: IPC endpoint when publishing (ignored with `--output-console`).
- `--batch-size`: Hailo inference batch size (default `1`).
- `--frame-stride`: process 1 in every N frames (default `1`).
- `--camera-index`: OpenCV camera index (default `0`).
- `--queue-maxsize`: internal frame queue size (default `4`).
- `--max-frames`: stop after N processed frames (useful for smoke tests).
- `--ipc-socket-type`: ZeroMQ socket type for publishing (default `PUB`).

Output processor (`output-processor`):
- `--ipc-endpoint` (required): IPC endpoint to bind/connect.
- `--bind` / `--no-bind`: bind (default) or connect to the endpoint.
- `--subscribe`: SUB filter prefix (default empty for all messages).
- `--max-messages`: stop after N messages (useful for smoke tests).
- `--ipc-socket-type`: ZeroMQ socket type for receiving (default `SUB`).

### Examples

Print CV detections from 10 frames directly to stdout (no IPC):

```bash
cv-processor --network /path/to/model.hef --output-console --max-frames 10
```

Use an IPC endpoint and a higher frame stride:

```bash
output-processor --ipc-endpoint ipc:///tmp/sundew-alt.ipc
cv-processor --network /path/to/model.hef --ipc-endpoint ipc:///tmp/sundew-alt.ipc --frame-stride 3
```

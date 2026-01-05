import json
from queue import Queue

from cv_processor.orchestrator import CvOrchestrator


class FakeCameraClient:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeInferClient:
    def __init__(self):
        self.runs = []
        self.closed = False

    def run(self, batch, callback):
        self.runs.append(batch)
        callback("ok", [{"detections": 1}])

    def close(self):
        self.closed = True


class FakeIpcClient:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send_json(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


def test_orchestrator_runs_pipeline_and_sends_messages():
    queue = Queue()
    queue.put("frame-1")
    queue.put("frame-2")

    camera = FakeCameraClient()
    infer = FakeInferClient()
    ipc = FakeIpcClient()

    orchestrator = CvOrchestrator(
        network="net.hef",
        batch_size=1,
        ipc_endpoint="ipc:///tmp/sundew.ipc",
        frame_queue=queue,
        camera_client=camera,
        infer_client=infer,
        ipc_client=ipc,
    )

    orchestrator.run(max_frames=2)

    assert camera.started is True
    assert camera.stopped is True
    assert infer.closed is True
    assert ipc.closed is True
    assert len(ipc.messages) == 2
    assert ipc.messages[0]["frame_id"] == 0
    assert ipc.messages[1]["frame_id"] == 1


def test_orchestrator_outputs_to_console_when_enabled(capsys):
    queue = Queue()
    queue.put("frame-1")

    camera = FakeCameraClient()
    infer = FakeInferClient()

    orchestrator = CvOrchestrator(
        network="net.hef",
        batch_size=1,
        ipc_endpoint=None,
        output_console=True,
        frame_queue=queue,
        camera_client=camera,
        infer_client=infer,
    )

    orchestrator.run(max_frames=1)

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["frame_id"] == 0
    assert payload["detections"] == [{"detections": 1}]

from queue import Queue

from cv_processor.camera_client import CameraClient


class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


def test_camera_client_enqueues_frames_and_releases():
    frames = ["f1", "f2"]
    capture = FakeCapture(frames)
    queue = Queue(maxsize=10)

    client = CameraClient(
        queue,
        camera_index=0,
        capture_factory=lambda _: capture,
    )
    client.start()
    client.stop(timeout=1.0)

    assert queue.qsize() == 2
    assert capture.released is True

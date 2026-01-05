from cv_processor.cli import main


class FakeOrchestrator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ran = False

    def run(self, *, max_frames=None):
        self.ran = True
        self.max_frames = max_frames


def test_main_smoke():
    args = [
        "--network",
        "net.hef",
        "--ipc-endpoint",
        "ipc:///tmp/sundew.ipc",
        "--max-frames",
        "1",
    ]
    assert main(args, orchestrator_cls=FakeOrchestrator) == 0

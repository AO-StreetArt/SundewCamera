import json
from unittest.mock import MagicMock

from output_processor.cli import main
from output_processor.orchestrator import OutputOrchestrator


def test_orchestrator_receives_and_prints_messages(capsys):
    messages = [
        {"frame_id": 1, "detections": []},
        {"frame_id": 2, "detections": [{"class": "cat"}]},
    ]
    ipc = MagicMock()
    ipc.recv_json.side_effect = list(messages)
    orchestrator = OutputOrchestrator(
        ipc_endpoint="ipc:///tmp/sundew.ipc",
        ipc_client=ipc,
    )

    orchestrator.run(max_messages=2)

    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line) for line in lines] == messages
    ipc.close.assert_called_once_with()
    assert ipc.recv_json.call_count == 2


def test_cli_constructs_orchestrator_and_runs():
    orchestrator_instance = MagicMock()
    orchestrator_cls = MagicMock(return_value=orchestrator_instance)

    exit_code = main(
        [
            "--ipc-endpoint",
            "ipc:///tmp/sundew.ipc",
            "--max-messages",
            "3",
        ],
        orchestrator_cls=orchestrator_cls,
    )

    assert exit_code == 0
    orchestrator_cls.assert_called_once_with(
        ipc_endpoint="ipc:///tmp/sundew.ipc",
        ipc_socket_type="SUB",
        bind=True,
        subscribe="",
    )
    orchestrator_instance.run.assert_called_once_with(max_messages=3)

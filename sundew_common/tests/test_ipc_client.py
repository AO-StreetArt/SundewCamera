from unittest.mock import MagicMock

from sundew_common.ipc_client import ZmqIpcClient


def test_ipc_client_connects_pub_socket():
    socket = MagicMock()
    context = MagicMock()
    context.socket.return_value = socket

    zmq_module = MagicMock()
    zmq_module.Context.instance.return_value = context
    zmq_module.PUB = 1

    client = ZmqIpcClient(
        "ipc:///tmp/sundew.ipc",
        "PUB",
        zmq_module=zmq_module,
    )

    context.socket.assert_called_once_with(1)
    socket.connect.assert_called_once_with("ipc:///tmp/sundew.ipc")
    socket.bind.assert_not_called()
    client.close()
    socket.close.assert_called_once()


def test_ipc_client_binds_sub_socket_with_subscription():
    socket = MagicMock()
    context = MagicMock()
    context.socket.return_value = socket

    zmq_module = MagicMock()
    zmq_module.Context.instance.return_value = context
    zmq_module.SUB = 2
    zmq_module.SUBSCRIBE = 3

    ZmqIpcClient(
        "ipc:///tmp/sundew.ipc",
        "SUB",
        bind=True,
        subscribe="",
        zmq_module=zmq_module,
    )

    context.socket.assert_called_once_with(2)
    socket.setsockopt_string.assert_called_once_with(3, "")
    socket.bind.assert_called_once_with("ipc:///tmp/sundew.ipc")
    socket.connect.assert_not_called()

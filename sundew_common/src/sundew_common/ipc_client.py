"""ZeroMQ IPC client helpers."""

from __future__ import annotations

from typing import Any, Optional


class ZmqIpcClient:
    """Create a ZeroMQ socket for IPC with minimal defaults."""

    def __init__(
        self,
        endpoint: str,
        socket_type: str,
        *,
        bind: bool = False,
        subscribe: Optional[str] = "",
        zmq_module: Optional[Any] = None,
    ) -> None:
        if zmq_module is None:
            try:
                import zmq as zmq_module  # type: ignore[no-redef]
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise ImportError(
                    "pyzmq is required. Install pyzmq or pass zmq_module."
                ) from exc

        socket_type_value = getattr(zmq_module, socket_type)
        context = zmq_module.Context.instance()
        self._socket = context.socket(socket_type_value)

        if socket_type == "SUB" and subscribe is not None:
            self._socket.setsockopt_string(zmq_module.SUBSCRIBE, subscribe)

        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)

    def __enter__(self) -> "ZmqIpcClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def send_json(self, message: Any) -> None:
        self._socket.send_json(message)

    def recv_json(self, *, flags: int = 0) -> Any:
        return self._socket.recv_json(flags=flags)

    def close(self) -> None:
        self._socket.close()

"""Processing orchestrator for output pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sundew_common.ipc_client import ZmqIpcClient

logger = logging.getLogger(__name__)


class OutputOrchestrator:
    """Receives IPC messages and prints them to stdout."""

    def __init__(
        self,
        *,
        ipc_endpoint: str,
        ipc_socket_type: str = "SUB",
        bind: bool = False,
        subscribe: Optional[str] = "",
        ipc_client: Optional[ZmqIpcClient] = None,
    ) -> None:
        self._ipc_client = ipc_client or ZmqIpcClient(
            ipc_endpoint,
            ipc_socket_type,
            bind=bind,
            subscribe=subscribe,
        )

    def run(self, *, max_messages: Optional[int] = None) -> None:
        """Receive IPC messages and print them to stdout."""
        processed = 0
        logger.info("Output orchestrator run loop started")
        try:
            while True:
                message = self._ipc_client.recv_json()
                print(json.dumps(message), flush=True)
                processed += 1
                if max_messages is not None and processed >= max_messages:
                    logger.info("Reached max_messages=%s, stopping", max_messages)
                    break
        finally:
            self._ipc_client.close()
            logger.info("Output orchestrator shut down")

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return repr(value)

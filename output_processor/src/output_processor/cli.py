"""CLI entry point for output-processor."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional, Type

from .orchestrator import OutputOrchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Sundew output processor."
    )
    parser.add_argument("--ipc-endpoint", required=True)
    parser.add_argument("--ipc-socket-type", default="SUB")
    parser.add_argument(
        "--bind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bind to the IPC endpoint instead of connecting.",
    )
    parser.add_argument(
        "--subscribe",
        default="",
        help="Subscription filter for SUB sockets.",
    )
    parser.add_argument("--max-messages", type=int, default=None)
    return parser


def main(
    argv: Optional[Iterable[str]] = None,
    orchestrator_cls: Type[OutputOrchestrator] = OutputOrchestrator,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    orchestrator = orchestrator_cls(
        ipc_endpoint=args.ipc_endpoint,
        ipc_socket_type=args.ipc_socket_type,
        bind=args.bind,
        subscribe=args.subscribe,
    )
    logging.getLogger(__name__).info(
        "Starting output orchestrator (ipc=%s, bind=%s, subscribe=%s)",
        args.ipc_endpoint,
        args.bind,
        args.subscribe,
    )
    orchestrator.run(max_messages=args.max_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry point for cv-processor."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional, Type

from .orchestrator import CvOrchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Sundew CV processor.")
    parser.add_argument("--network", required=True, help="Path to the Hailo HEF.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ipc-endpoint")
    parser.add_argument("--ipc-socket-type", default="PUB")
    parser.add_argument(
        "--output-console",
        action="store_true",
        help="Print detection messages to stdout instead of IPC.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Only process 1 in every N frames.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--queue-maxsize", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(
    argv: Optional[Iterable[str]] = None,
    orchestrator_cls: Type[CvOrchestrator] = CvOrchestrator,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.output_console and not args.ipc_endpoint:
        parser.error("--ipc-endpoint is required unless --output-console is set")

    orchestrator = orchestrator_cls(
        network=args.network,
        batch_size=args.batch_size,
        ipc_endpoint=args.ipc_endpoint,
        ipc_socket_type=args.ipc_socket_type,
        camera_index=args.camera_index,
        queue_maxsize=args.queue_maxsize,
        output_console=args.output_console,
        frame_stride=args.frame_stride,
    )
    logging.getLogger(__name__).info(
        "Starting CV orchestrator (network=%s, ipc=%s, output_console=%s)",
        args.network,
        args.ipc_endpoint,
        args.output_console,
    )
    orchestrator.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

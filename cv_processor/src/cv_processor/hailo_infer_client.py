"""Thin wrapper around HailoInfer for dependency injection and testability."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Type

logger = logging.getLogger(__name__)


class HailoInferClient:
    """Minimal client wrapper for HailoInfer."""

    def __init__(
        self,
        network: str,
        batch_size: int,
        infer_cls: Optional[Type[object]] = None,
    ) -> None:
        if infer_cls is None:
            try:
                from hailo_apps.python.core.common.hailo_inference import HailoInfer
            except ImportError as exc:  # pragma: no cover - exercised by runtime use
                raise ImportError(
                    "HailoInfer not available. Install hailo-apps or pass infer_cls."
                ) from exc
            infer_cls = HailoInfer

        self._infer = infer_cls(network, batch_size)
        logger.info("Initialized HailoInfer client (network=%s, batch_size=%s)", network, batch_size)

    def get_input_shape(self) -> tuple[int, int, int]:
        return self._infer.get_input_shape()

    def run(self, preprocessed_batch, callback: Callable) -> None:
        self._infer.run(preprocessed_batch, callback)

    def close(self) -> None:
        self._infer.close()
        logger.info("Closed HailoInfer client")

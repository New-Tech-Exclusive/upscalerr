from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def shutdown_runtime(app) -> None:
    """Ordered teardown: stop threads, sync CUDA, release GL."""
    try:
        app.stop()
    except Exception as exc:
        logger.warning("Error during app.stop(): %s", exc)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("CUDA cleanup error: %s", exc)

    logger.info("Runtime shutdown complete")

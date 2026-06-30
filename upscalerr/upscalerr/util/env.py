"""Configure process environment before importing PyTorch / MKL on Windows."""

from __future__ import annotations

import os
import sys


def configure_runtime_env() -> None:
    """
    PyTorch, NumPy, and Intel MKL each ship libiomp5md.dll on Windows.
    Importing more than one copy aborts unless this flag is set first.
    """
    if sys.platform == "win32":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


configure_runtime_env()

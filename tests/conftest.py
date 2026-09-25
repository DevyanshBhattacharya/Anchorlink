"""Test configuration.

**Why the suite is split.** LightGBM, scikit-learn, PyTorch and sparse_dot_topn
each ship their own OpenMP runtime.  Loading two of them into one process is
pathological on macOS: at best the sparse top-k product silently drops to a
single thread (measured: a 25x slowdown, PROGRESS.md §7), at worst the process
deadlocks inside ``kmp_flag_64::wait``.  Tests that import torch are therefore
marked ``torch`` and run in their own process:

    python -m pytest tests -q -m "not torch"
    python -m pytest tests -q -m torch
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "business_entity_resolution" / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torch: needs PyTorch; run in a separate process (see this file)")

"""Processing-latency measurement with a monotonic clock.

Processing milliseconds (this module) are NOT detection delay (minutes after onset) and NOT advance
warning (minutes before onset). A faster model is not a model that warns earlier.
"""

from __future__ import annotations

import platform
import time
from contextlib import contextmanager

import numpy as np
import torch


class Timer:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.samples.setdefault(name, []).append((time.perf_counter_ns() - t0) / 1e6)

    def summary(self, warmup: int = 0, batch_size: int | None = None) -> dict:
        out = {}
        for name, vals in self.samples.items():
            v = np.asarray(vals[warmup:], dtype=float)
            if not len(v):
                continue
            out[name] = {
                "p50_ms": float(np.percentile(v, 50)),
                "p95_ms": float(np.percentile(v, 95)),
                "p99_ms": float(np.percentile(v, 99)),
                "n_measured": int(len(v)),
                "n_warmup_discarded": int(min(warmup, len(vals))),
                "batch_size": batch_size,
            }
        return out


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "clock": "time.perf_counter_ns (monotonic)",
    }

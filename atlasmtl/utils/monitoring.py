from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

import torch

try:  # pragma: no cover
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

T = TypeVar("T")


def resolve_show_progress(show_progress: Optional[bool]) -> bool:
    if show_progress is not None:
        return show_progress
    return sys.stderr.isatty()


def resolve_show_summary(show_summary: Optional[bool]) -> bool:
    if show_summary is not None:
        return show_summary
    return sys.stdout.isatty()


def progress_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str,
    show_progress: Optional[bool],
) -> Iterable[T]:
    enabled = resolve_show_progress(show_progress)
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def _max_rss_gb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(max_rss / (1024 ** 3), 4)
    return round((max_rss * 1024) / (1024 ** 3), 4)


@dataclass
class RuntimeMonitor:
    phase: str
    device: torch.device

    def __post_init__(self) -> None:
        self.start_time = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def finish(self, *, num_items: int, num_batches: int) -> dict[str, object]:
        elapsed_seconds = time.perf_counter() - self.start_time
        summary = {
            "phase": self.phase,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "num_items": int(num_items),
            "num_batches": int(num_batches),
            "items_per_second": round(num_items / elapsed_seconds, 4) if elapsed_seconds > 0 else None,
            "process_peak_rss_gb": _max_rss_gb(),
        }
        if self.device.type == "cuda":
            summary["gpu_peak_memory_gb"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 4
            )
        else:
            summary["gpu_peak_memory_gb"] = None
        return summary

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, TypeVar

import torch

try:  # pragma: no cover
    import psutil
except Exception:  # pragma: no cover
    psutil = None

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


def _bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024 ** 3), 4)


def _safe_children(process: "psutil.Process") -> list["psutil.Process"]:
    try:
        return process.children(recursive=True)
    except Exception:
        return []


def _safe_memory_rss(process: "psutil.Process") -> int:
    try:
        return int(process.memory_info().rss)
    except Exception:
        return 0


def _safe_cpu_percent(process: "psutil.Process") -> float:
    try:
        return float(process.cpu_percent(interval=None))
    except Exception:
        return 0.0


def _process_tree(process_id: int) -> list["psutil.Process"]:
    if psutil is None:
        return []
    try:
        root = psutil.Process(process_id)
    except Exception:
        return []
    processes = [root]
    processes.extend(_safe_children(root))
    return processes


def _parse_time_v_maxrss_gb(path: str) -> float | None:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size (kbytes):"):
            value = line.split(":", 1)[1].strip()
            try:
                kb = float(value)
            except Exception:
                return None
            return round((kb * 1024.0) / (1024 ** 3), 4)
    return None


@dataclass
class SampledResourceTracker:
    device: torch.device | str
    process_id: int | None = None
    sample_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.process_id = int(self.process_id or os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_sum_bytes = 0
        self._rss_samples = 0
        self._rss_peak_bytes = 0
        self._cpu_percent_sum = 0.0
        self._cpu_percent_samples = 0
        self._gpu_memory_sum_bytes = 0
        self._gpu_memory_samples = 0
        self._gpu_peak_bytes = 0
        self.start_time = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._prime_cpu_percent()
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="atlasmtl-resource-tracker", daemon=True)
        self._thread.start()

    def finish(
        self,
        *,
        phase: str,
        num_items: int,
        num_batches: int | None = None,
        device_used: str | None = None,
        num_threads_used: int | None = None,
    ) -> dict[str, object]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval_seconds * 2, 0.2))
        self._sample_once()
        elapsed_seconds = time.perf_counter() - self.start_time
        process_peak_rss_gb = _bytes_to_gb(self._rss_peak_bytes)
        if process_peak_rss_gb is None and self.process_id == os.getpid():
            process_peak_rss_gb = _max_rss_gb()
        gpu_peak_gb = _bytes_to_gb(self._gpu_peak_bytes)
        if gpu_peak_gb is None and self.device.type == "cuda" and torch.cuda.is_available():
            gpu_peak_gb = round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 4)
        cpu_percent_avg = (
            round(self._cpu_percent_sum / self._cpu_percent_samples, 4)
            if self._cpu_percent_samples
            else None
        )
        return {
            "phase": phase,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "num_items": int(num_items),
            "num_batches": int(num_batches) if num_batches is not None else None,
            "rss_samples": int(self._rss_samples),
            "cpu_samples": int(self._cpu_percent_samples),
            "gpu_samples": int(self._gpu_memory_samples),
            "items_per_second": round(num_items / elapsed_seconds, 4) if elapsed_seconds > 0 else None,
            "process_peak_rss_gb": process_peak_rss_gb,
            "process_avg_rss_gb": _bytes_to_gb(self._rss_sum_bytes / self._rss_samples) if self._rss_samples else None,
            "cpu_percent_avg": cpu_percent_avg,
            "cpu_core_equiv_avg": round(cpu_percent_avg / 100.0, 4) if cpu_percent_avg is not None else None,
            "gpu_peak_memory_gb": gpu_peak_gb,
            "gpu_avg_memory_gb": _bytes_to_gb(self._gpu_memory_sum_bytes / self._gpu_memory_samples)
            if self._gpu_memory_samples
            else None,
            "device_used": device_used or self.device.type,
            "num_threads_used": num_threads_used,
        }

    def _prime_cpu_percent(self) -> None:
        if psutil is None:
            return
        for process in _process_tree(self.process_id):
            _safe_cpu_percent(process)

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        processes = _process_tree(self.process_id)
        if processes:
            rss_total = sum(_safe_memory_rss(process) for process in processes)
            self._rss_sum_bytes += rss_total
            self._rss_samples += 1
            self._rss_peak_bytes = max(self._rss_peak_bytes, rss_total)

            cpu_total = sum(_safe_cpu_percent(process) for process in processes)
            self._cpu_percent_sum += cpu_total
            self._cpu_percent_samples += 1
        elif self.process_id == os.getpid():
            rss_bytes = int(round(_max_rss_gb() * (1024 ** 3)))
            self._rss_sum_bytes += rss_bytes
            self._rss_samples += 1
            self._rss_peak_bytes = max(self._rss_peak_bytes, rss_bytes)

        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_bytes = int(torch.cuda.memory_allocated(self.device))
            self._gpu_memory_sum_bytes += gpu_bytes
            self._gpu_memory_samples += 1
            self._gpu_peak_bytes = max(self._gpu_peak_bytes, gpu_bytes)


def run_subprocess_monitored(
    command: list[str],
    *,
    cwd: str | os.PathLike[str] | None,
    env: dict[str, str] | None,
    phase: str,
    n_items: int,
    device: str = "cpu",
    num_threads_used: int | None = None,
    text: bool = True,
    capture_output: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    time_file: str | None = None
    wrapped_command = list(command)
    if psutil is None and text and capture_output and os.path.exists("/usr/bin/time"):
        fd, time_file = tempfile.mkstemp(prefix="atlasmtl_time_", suffix=".txt")
        os.close(fd)
        wrapped_command = ["/usr/bin/time", "-v", "-o", time_file, "--", *command]

    process = subprocess.Popen(
        wrapped_command,
        cwd=cwd,
        env=env,
        text=text,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )
    tracker = SampledResourceTracker(device=device, process_id=process.pid)
    tracker.start()
    stdout, stderr = process.communicate()
    usage = tracker.finish(
        phase=phase,
        num_items=n_items,
        num_batches=None,
        device_used=device,
        num_threads_used=num_threads_used,
    )
    if time_file:
        parsed_rss_gb = _parse_time_v_maxrss_gb(time_file)
        if parsed_rss_gb is not None:
            current_peak = usage.get("process_peak_rss_gb")
            current_avg = usage.get("process_avg_rss_gb")
            if current_peak in (None, 0.0):
                usage["process_peak_rss_gb"] = parsed_rss_gb
            if current_avg in (None, 0.0):
                usage["process_avg_rss_gb"] = parsed_rss_gb
        try:
            os.remove(time_file)
        except Exception:
            pass
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, usage


@dataclass
class RuntimeMonitor:
    phase: str
    device: torch.device

    def __post_init__(self) -> None:
        self._tracker = SampledResourceTracker(device=self.device)
        self._tracker.start()

    def finish(self, *, num_items: int, num_batches: int) -> dict[str, object]:
        return self._tracker.finish(
            phase=self.phase,
            num_items=num_items,
            num_batches=num_batches,
            device_used=self.device.type,
            num_threads_used=None,
        )

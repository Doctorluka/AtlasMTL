from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _safe_run(cmd: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
        return {
            "cmd": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "cmd": cmd,
            "error": repr(exc),
        }


def collect_cuda_report() -> dict[str, object]:
    report: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env_subset": {
            key: os.environ.get(key)
            for key in [
                "CONDA_DEFAULT_ENV",
                "CONDA_PREFIX",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "PATH",
            ]
        },
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "torch_version_cuda": torch.version.cuda,
            "cuda_is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "cudnn_available": torch.backends.cudnn.is_available(),
        }

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            report["torch"]["device_0"] = {
                "name": device_name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "major": props.major,
                "minor": props.minor,
            }
            try:
                x = torch.randn((1024, 1024), device="cuda")
                y = torch.randn((1024, 1024), device="cuda")
                z = (x @ y).sum().item()
                report["torch"]["cuda_matmul_test"] = {
                    "ok": True,
                    "result_sum": z,
                }
            except Exception as exc:
                report["torch"]["cuda_matmul_test"] = {
                    "ok": False,
                    "error": repr(exc),
                }
        else:
            report["torch"]["cuda_matmul_test"] = {
                "ok": False,
                "error": "CUDA not available in this process",
            }
    except Exception as exc:
        report["torch_import_error"] = repr(exc)

    report["system_checks"] = {
        "nvidia_smi": _safe_run(["nvidia-smi"]),
        "which_python": _safe_run(["which", "python"]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CUDA runtime diagnostics for the current Python process."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    report = collect_cuda_report()
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

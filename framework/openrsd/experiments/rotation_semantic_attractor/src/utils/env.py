from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def git_commit(cwd: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception as exc:  # pragma: no cover - depends on checkout state
        return f"UNKNOWN:{type(exc).__name__}"


def collect_env(cwd: str | Path) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(cwd),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
    }
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": str(torch.version.cuda),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
            }
        )
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


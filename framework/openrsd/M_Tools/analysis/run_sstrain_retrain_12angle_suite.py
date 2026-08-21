#!/usr/bin/env python3
"""Run ss_train retraining and 12-angle no-TTA evaluation for DOTA1 models.

This script is intentionally conservative: it only trains models whose local
config can be matched to the requested experiment without inventing missing
schedule or augmentation settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ANGLES = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]

MODEL_ORDER = ["redet", "retinanet_msrr", "retinanet_amp", "rtmdet_l", "h2rbox_v2", "h2rbox"]

DISPLAY_NAMES = {
    "h2rbox": "H2RBox 3xMS",
    "rtmdet_l": "RTMDet-L 3xMS",
    "h2rbox_v2": "H2RBox-v2 1xMS+RR",
    "retinanet_amp": "RetinaNet R50 AMP 1x",
    "retinanet_msrr": "RetinaNet R50 MS+RR 1x",
    "redet": "ReDet Re50 MS+RR",
}

SUMMARY_ORDER = ["h2rbox", "rtmdet_l", "h2rbox_v2", "retinanet_amp", "retinanet_msrr", "redet"]

OFFICIAL_BASELINE = {
    "h2rbox": {
        "000": 0.8084, "030": 0.7476, "060": 0.7309, "090": 0.7544,
        "120": 0.7329, "150": 0.7465, "180": 0.8113, "210": 0.7489,
        "240": 0.7323, "270": 0.7581, "300": 0.7323, "330": 0.7473,
        "mean": 0.7542, "range": 0.0804,
    },
    "rtmdet_l": {
        "000": 0.9183, "030": 0.8951, "060": 0.8924, "090": 0.9161,
        "120": 0.8853, "150": 0.8827, "180": 0.9170, "210": 0.8828,
        "240": 0.8853, "270": 0.9102, "300": 0.8914, "330": 0.8971,
        "mean": 0.8978, "range": 0.0356,
    },
    "h2rbox_v2": {
        "000": 0.8148, "030": 0.8068, "060": 0.8098, "090": 0.8117,
        "120": 0.7993, "150": 0.8032, "180": 0.8047, "210": 0.7959,
        "240": 0.8037, "270": 0.8100, "300": 0.8055, "330": 0.8120,
        "mean": 0.8064, "range": 0.0189,
    },
    "retinanet_amp": {
        "000": 0.4446, "030": 0.3580, "060": 0.3493, "090": 0.4140,
        "120": 0.3464, "150": 0.3428, "180": 0.4444, "210": 0.3580,
        "240": 0.3398, "270": 0.4154, "300": 0.3422, "330": 0.3574,
        "mean": 0.3760, "range": 0.1048,
    },
    "retinanet_msrr": {
        "000": 0.4474, "030": 0.3639, "060": 0.3749, "090": 0.4823,
        "120": 0.3865, "150": 0.3716, "180": 0.4503, "210": 0.3553,
        "240": 0.3734, "270": 0.4807, "300": 0.3873, "330": 0.3765,
        "mean": 0.4042, "range": 0.1270,
    },
    "redet": {
        "000": 0.7591, "030": 0.6979, "060": 0.6867, "090": 0.7517,
        "120": 0.6853, "150": 0.6865, "180": 0.7524, "210": 0.6881,
        "240": 0.6863, "270": 0.7489, "300": 0.6960, "330": 0.6871,
        "mean": 0.7105, "range": 0.0738,
    },
}

FORBIDDEN_CKPT_WORDS = (
    "trainval", "train+val", "dota", "DOTA", "ss_train_retrain", "results", "work_dirs",
    "epoch_", "best_", "latest.pth", "redet", "retinanet", "h2rbox", "rtmdet",
)

ALLOWED_PRETRAIN_MARKERS = (
    "torchvision://", "open-mmlab://", "detectron2://", "mmcls://", "coco", "COCO",
    "imagenet", "ImageNet", "in1k", "pretrain", "Pretrain", "cspnext", "resnet50",
    "re_resnet50_c8_batch256",
)

DOTA_CLASSES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle",
    "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank",
    "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter",
)


@dataclass
class ModelSpec:
    key: str
    display: str
    expected: str
    primary_config: str | None
    status: str = "PENDING"
    blocked_reason: str = ""
    candidates: list[str] = field(default_factory=list)
    generated_config: str = ""
    smoke_config: str = ""
    latest_checkpoint: str = ""
    best_checkpoint: str = ""
    train_log: str = ""
    load_from: str = ""
    pretrain_type: str = "unknown"
    forbidden_dota_ckpt_used: bool = False
    batch_size_per_gpu: int | None = None
    global_batch_size: int | None = None
    lr: str = ""
    amp: bool = False
    ms_aug: bool = False
    rr_aug: bool = False
    final_epoch_or_iter: str = ""
    original_config_abs: str = ""


def now() -> str:
    return datetime.now().strftime("%F %T")


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def shell_join(argv: Iterable[Any]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in argv)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rtk_env_prefix(repo_root: Path, gpu_ids: str) -> list[str]:
    return [
        "rtk",
        "env",
        "PYTHONNOUSERSITE=1",
        "MPLCONFIGDIR=/tmp/mplconfig",
        f"PYTHONPATH={repo_root}:{repo_root / 'tools'}",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        f"CUDA_VISIBLE_DEVICES={gpu_ids}",
    ]


def command_string(repo_root: Path, gpu_ids: str, argv: list[Any]) -> str:
    return shell_join([*rtk_env_prefix(repo_root, gpu_ids), *argv])


def quick_run(argv: list[Any], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [str(x) for x in argv],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, "", repr(exc)


def nvidia_smi(repo_root: Path) -> str:
    rc, out, err = quick_run(["rtk", "nvidia-smi"], repo_root, timeout=20)
    return out if rc == 0 else err


def detect_error(text: str) -> str:
    low = text.lower()
    patterns = [
        ("timeout: command exceeded timeout", "FAILED_TIMEOUT"),
        ("cuda out of memory", "FAILED_OOM"),
        ("outofmemoryerror", "FAILED_OOM"),
        ("nccl", "FAILED_NCCL"),
        ("init_process_group", "FAILED_DDP_INIT"),
        ("no such file or directory", "FAILED_FILE_MISSING"),
        ("filenotfounderror", "FAILED_FILE_MISSING"),
        ("there is no txt file", "FAILED_ANNOTATION_MISSING"),
        ("class", "FAILED_CLASS_OR_FORMAT"),
        ("gt_instances", "FAILED_DUMPDET_GT_INSTANCES"),
        ("prediction", "FAILED_PREDICTION"),
        (r"\bnan\b", "FAILED_NAN"),
        ("traceback", "FAILED_TRACEBACK"),
        ("testtimeaug", "FAILED_TTA_RISK"),
    ]
    for needle, label in patterns:
        if needle.startswith("\\"):
            if re.search(needle, low):
                return label
        elif needle in low:
            return label
    return "FAILED" if text.strip() else ""


class CommandLogger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logs_root = args.work_dir / "logs"
        self.commands_jsonl = self.logs_root / "commands.jsonl"

    def version_info(self, gpu_ids: str) -> dict[str, Any]:
        code = (
            "import sys, json, importlib\n"
            "mods=['torch','mmcv','mmengine','mmdet','mmrotate']\n"
            "out={'python_executable': sys.executable}\n"
            "for m in mods:\n"
            "    try:\n"
            "        mod=importlib.import_module(m)\n"
            "        out[m]=getattr(mod,'__version__','unknown')\n"
            "    except Exception as e:\n"
            "        out[m]='ERR:'+repr(e)\n"
            "try:\n"
            "    import torch\n"
            "    out['torch_cuda']=getattr(torch.version,'cuda',None)\n"
            "except Exception: pass\n"
            "print(json.dumps(out))\n"
        )
        rc, out, err = quick_run([*rtk_env_prefix(self.args.repo_root, gpu_ids), self.args.python_bin, "-c", code], self.args.repo_root)
        if rc == 0:
            try:
                return json.loads(out.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                pass
        return {"python_executable": sys.executable, "version_error": err or out}

    def run(
        self,
        task: str,
        argv: list[Any],
        gpu_ids: str,
        timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ts = run_stamp()
        log_dir = self.logs_root / task / ts
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        before_path = log_dir / "nvidia_smi_before.txt"
        after_path = log_dir / "nvidia_smi_after.txt"
        meta_path = log_dir / "meta.json"
        before_text = nvidia_smi(self.args.repo_root)
        write_text(before_path, before_text)
        start = now()
        t0 = time.time()
        full_argv = [*rtk_env_prefix(self.args.repo_root, gpu_ids), *map(str, argv)]
        if extra_env:
            full_argv = [*rtk_env_prefix(self.args.repo_root, gpu_ids), *[f"{k}={v}" for k, v in extra_env.items()], *map(str, argv)]
        command = shell_join(full_argv)
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            out.write(f"[{start}] command={command}\n")
            out.write(f"cwd={self.args.repo_root}\n\n")
            out.flush()
            try:
                proc = subprocess.Popen(
                    full_argv,
                    cwd=str(self.args.repo_root),
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=True,
                )
                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    rc = 124
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        time.sleep(3)
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    err.write("\nTIMEOUT: command exceeded timeout\n")
            except subprocess.TimeoutExpired:
                rc = 124
                err.write("\nTIMEOUT: command exceeded timeout\n")
        end = now()
        after_text = nvidia_smi(self.args.repo_root)
        write_text(after_path, after_text)
        stdout_tail = "\n".join(read_text(stdout_path).splitlines()[-150:])
        stderr_tail = "\n".join(read_text(stderr_path).splitlines()[-150:])
        git_rc, git_out, git_err = quick_run(["rtk", "git", "rev-parse", "HEAD"], self.args.repo_root)
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        meta = {
            "task": task,
            "command": command,
            "start_time": start,
            "end_time": end,
            "return_code": rc,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "nvidia_smi_before": str(before_path),
            "nvidia_smi_after": str(after_path),
            "git_commit_hash": git_out.strip() if git_rc == 0 else git_err.strip(),
            "conda_env": conda_env,
            "versions": self.version_info(gpu_ids),
            "duration_sec": time.time() - t0,
            "error_class": detect_error(stdout_tail + "\n" + stderr_tail) if rc else "",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        write_json(meta_path, meta)
        append_jsonl(self.commands_jsonl, meta)
        return meta


def import_config() -> Any:
    repo_tools = Path(__file__).resolve().parents[2] / "tools"
    if str(repo_tools) not in sys.path:
        sys.path.insert(0, str(repo_tools))
    try:
        from openrsd_env import preload_installed_mmengine

        preload_installed_mmengine()
    except Exception:
        pass
    from mmengine.config import Config

    return Config


def find_configs(repo_root: Path) -> dict[str, ModelSpec]:
    rg_cmd = ["rtk", "rg", "--files"]
    rc, out, _ = quick_run(rg_cmd, repo_root, timeout=60)
    files = [line.strip() for line in out.splitlines() if line.strip().endswith(".py")] if rc == 0 else []

    def candidates_for(*needles: str) -> list[str]:
        lowered = [(f, f.lower()) for f in files]
        hits = []
        for f, low in lowered:
            if all(n.lower() in low for n in needles):
                hits.append(f)
        return sorted(hits)

    specs = {
        "redet": ModelSpec(
            key="redet",
            display=DISPLAY_NAMES["redet"],
            expected="redet re50 refpn ms rr dota",
            primary_config="mmrotate_configs/redet/redet-le90_re50_refpn_rr-1x_dota-ms.py",
        ),
        "retinanet_msrr": ModelSpec(
            key="retinanet_msrr",
            display=DISPLAY_NAMES["retinanet_msrr"],
            expected="rotated retinanet r50 ms rr 1x dota",
            primary_config="mmrotate_configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_rr-1x_dota-ms.py",
        ),
        "retinanet_amp": ModelSpec(
            key="retinanet_amp",
            display=DISPLAY_NAMES["retinanet_amp"],
            expected="rotated retinanet r50 amp 1x dota",
            primary_config="mmrotate_configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py",
        ),
        "rtmdet_l": ModelSpec(
            key="rtmdet_l",
            display=DISPLAY_NAMES["rtmdet_l"],
            expected="rotated rtmdet l coco pretrain 3x dota ms",
            primary_config="mmrotate_configs/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py",
        ),
        "h2rbox_v2": ModelSpec(
            key="h2rbox_v2",
            display=DISPLAY_NAMES["h2rbox_v2"],
            expected="h2rbox v2 1x ms rr dota training config",
            primary_config=None,
        ),
        "h2rbox": ModelSpec(
            key="h2rbox",
            display=DISPLAY_NAMES["h2rbox"],
            expected="h2rbox 3x ms dota training config",
            primary_config=None,
        ),
    }
    specs["redet"].candidates = candidates_for("redet", "dota")
    specs["retinanet_msrr"].candidates = candidates_for("retinanet", "r50", "dota")
    specs["retinanet_amp"].candidates = specs["retinanet_msrr"].candidates
    specs["rtmdet_l"].candidates = candidates_for("rtmdet", "l", "dota")
    specs["h2rbox_v2"].candidates = candidates_for("h2rbox") + candidates_for("h2rbox_v2")
    specs["h2rbox"].candidates = candidates_for("h2rbox", "dota")

    h2_3x = repo_root / "mmrotate_configs/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dota.py"
    h2_ms = repo_root / "mmrotate_configs/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota-ms.py"
    if h2_3x.exists() and h2_ms.exists():
        specs["h2rbox"].status = "BLOCKED"
        specs["h2rbox"].blocked_reason = (
            "No exact H2RBox 3xMS training config found locally. "
            "Found 3x non-MS and 1xMS configs, but combining them would invent a config."
        )
    elif h2_3x.exists():
        specs["h2rbox"].status = "BLOCKED"
        specs["h2rbox"].blocked_reason = "Only H2RBox 3x non-MS config found; requested 3xMS config is missing."
    else:
        specs["h2rbox"].status = "BLOCKED"
        specs["h2rbox"].blocked_reason = "H2RBox 3xMS config missing."

    h2v2_eval = repo_root / "M_configs/RotationStudy/h2rbox_v2_r50_fpn_dota1_ms_rr_eval.py"
    if h2v2_eval.exists():
        specs["h2rbox_v2"].candidates.insert(0, str(h2v2_eval.relative_to(repo_root)))
        specs["h2rbox_v2"].status = "BLOCKED"
        specs["h2rbox_v2"].blocked_reason = (
            "Only H2RBox-v2 eval adapter found locally; it sets train_dataloader/train_cfg/"
            "optim_wrapper/param_scheduler to None, so the training config is missing."
        )
    else:
        specs["h2rbox_v2"].status = "BLOCKED"
        specs["h2rbox_v2"].blocked_reason = "H2RBox-v2 training config missing."

    for spec in specs.values():
        if spec.primary_config:
            path = repo_root / spec.primary_config
            if path.exists():
                spec.status = "CONFIG_FOUND"
                spec.original_config_abs = str(path)
            else:
                spec.status = "BLOCKED"
                spec.blocked_reason = f"Expected config missing: {spec.primary_config}"
    return specs


def collect_paths(obj: Any, paths: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"data_root", "ann_file", "img_path", "checkpoint", "load_from", "resume_from"} and isinstance(v, str):
                paths.append(v)
            collect_paths(v, paths)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            collect_paths(item, paths)


def collect_checkpoint_refs(obj: Any, refs: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"load_from", "resume_from", "checkpoint"} and isinstance(v, str):
                refs.append(v)
            collect_checkpoint_refs(v, refs)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            collect_checkpoint_refs(item, refs)


def classify_pretrain(refs: list[str]) -> tuple[str, bool, str]:
    if not refs:
        return "random init", False, ""
    forbidden = False
    chosen = refs[0]
    types = []
    for ref in refs:
        low = ref.lower()
        allowed = any(marker.lower() in low for marker in ALLOWED_PRETRAIN_MARKERS)
        has_forbidden = any(word.lower() in low for word in FORBIDDEN_CKPT_WORDS)
        if has_forbidden and not allowed:
            forbidden = True
        if "coco" in low:
            types.append("COCO pretrain")
        elif "imagenet" in low or "in1k" in low or "torchvision://" in low:
            types.append("ImageNet pretrain")
        elif "pretrain" in low or "re_resnet50" in low:
            types.append("backbone pretrain")
    return ", ".join(sorted(set(types))) or "checkpoint/init_cfg pretrain", forbidden, chosen


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def bool_contains(obj: Any, token: str) -> bool:
    return token.lower() in repr(obj).lower()


def config_has_tta(cfg: Any) -> bool:
    return any(t in repr(cfg.get("test_pipeline", "")) + repr(cfg.get("tta_pipeline", "")) + repr(cfg.get("test_dataloader", "")) for t in ("TestTimeAug", "MultiScaleFlipAug", "tta"))


def angle_dir(angle_root: Path, angle: str, train_root: Path) -> Path:
    direct = angle_root / f"angle_{angle}"
    if direct.is_dir():
        return direct
    base = train_root.parent
    matches = sorted(base.rglob(f"angle_{angle}"))
    if matches:
        return matches[0]
    return direct


def generate_config_text(
    *,
    original: Path,
    train_root: Path,
    angle000: Path,
    batch_size: int | None,
    num_workers: int,
    work_dir: Path,
) -> str:
    batch_line = f"batch_size={batch_size}," if batch_size else ""
    return f"""# Auto-generated by run_sstrain_retrain_12angle_suite.py
_base_ = r'{original}'

train_data_root = r'{train_root}/'
angle000_data_root = r'{angle000}/'

load_from = None
resume = False
work_dir = r'{work_dir}'

train_dataloader = dict(
    _delete_=True,
    {batch_line}
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=train_data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{{{_base_.train_pipeline}}}}))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=angle000_data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{{{_base_.val_pipeline}}}}))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
"""


def generate_smoke_config_text(base_config: Path, max_iters: int, max_images: int, batch_size: int, num_workers: int, work_dir: Path) -> str:
    return f"""# Auto-generated smoke config.
_base_ = r'{base_config}'

work_dir = r'{work_dir}'

train_dataloader = dict(
    batch_size={batch_size},
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    dataset=dict(indices={max_images}))

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={max_iters + 1000})
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval={max(1, max_iters)}, by_epoch=False, max_keep_ckpts=2))
"""


def generate_angle_eval_config_text(
    base_config: Path,
    angle_root: Path,
    angle: str,
    num_workers: int,
    work_dir: Path,
    indices: int | None = None,
) -> str:
    indices_line = f"        indices={indices},\n" if indices is not None else ""
    return f"""# Auto-generated eval config for angle {angle}, no TTA.
_base_ = r'{base_config}'

angle_data_root = r'{angle_root}/'
work_dir = r'{work_dir}'
load_from = None
resume = False
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=angle_data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        img_shape=(1024, 1024),
        test_mode=True,
{indices_line}        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{{{_base_.val_pipeline}}}}))
val_dataloader = test_dataloader
test_evaluator = dict(type='DOTAMetric', metric='mAP')
val_evaluator = test_evaluator
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')
"""


def generate_configs(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> None:
    Config = import_config()
    cfg_dir = args.work_dir / "generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    angle000 = angle_dir(args.angle_root, "000", args.train_root)
    for spec in specs.values():
        if spec.status != "CONFIG_FOUND":
            continue
        original = Path(spec.original_config_abs)
        original_bs = 1
        try:
            original_cfg = Config.fromfile(str(original))
            original_bs = int(cfg_get(cfg_get(original_cfg, "train_dataloader", {}), "batch_size", 1) or 1)
        except Exception:
            original_bs = 1
        effective_bs = args.batch_size or original_bs
        out = cfg_dir / f"{spec.key}_ss_train.py"
        text = generate_config_text(
            original=original,
            train_root=args.train_root,
            angle000=angle000,
            batch_size=effective_bs,
            num_workers=args.num_workers,
            work_dir=args.work_dir / "train" / spec.key,
        )
        write_text(out, text)
        spec.generated_config = str(out)
        try:
            cfg = Config.fromfile(str(out))
            refs: list[str] = []
            collect_checkpoint_refs(cfg.to_dict(), refs)
            spec.pretrain_type, spec.forbidden_dota_ckpt_used, spec.load_from = classify_pretrain(refs)
            spec.amp = "AmpOptimWrapper" in repr(cfg.get("optim_wrapper", ""))
            spec.ms_aug = bool_contains(cfg.get("train_pipeline", ""), "Resize") and "ms" in spec.primary_config.lower()
            spec.rr_aug = bool_contains(cfg.get("train_pipeline", ""), "RandomRotate") or "rr" in spec.primary_config.lower()
            bs = cfg_get(cfg_get(cfg, "train_dataloader", {}), "batch_size", args.batch_size)
            spec.batch_size_per_gpu = int(bs) if bs else args.batch_size
            ngpu = max(1, len(args.gpu_ids.split(",")))
            spec.global_batch_size = spec.batch_size_per_gpu * ngpu
            optim = cfg.get("optim_wrapper", {})
            spec.lr = str(cfg_get(cfg_get(optim, "optimizer", {}), "lr", ""))
            spec.status = "GENERATED"
            if spec.forbidden_dota_ckpt_used:
                spec.status = "FAILED_FOR_LEAKAGE_RISK"
                spec.blocked_reason = "Forbidden DOTA-like checkpoint detected in load_from/init_cfg."
        except Exception as exc:  # noqa: BLE001
            spec.status = "BLOCKED"
            spec.blocked_reason = f"Generated config failed to load: {exc!r}"


def dataset_inventory(train_root: Path, angle_root_arg: Path) -> dict[str, Any]:
    img_dir = train_root / "images"
    ann_dir = train_root / "annfiles"
    label_dir = train_root / "labelTxt"
    step6_dir = train_root / "Step6_Format_labels"
    ann_candidates = [p for p in [ann_dir, label_dir, step6_dir, train_root / "annotations"] if p.exists()]
    ann_source = ann_candidates[0] if ann_candidates else ann_dir
    image_files = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.tif"), *img_dir.glob("*.bmp")]) if img_dir.exists() else []
    ann_files = sorted(ann_source.glob("*.txt")) if ann_source.exists() else []
    cls_counts = {c: 0 for c in DOTA_CLASSES}
    parse_errors = []
    for txt in ann_files:
        try:
            for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 9 and parts[8] in cls_counts:
                    cls_counts[parts[8]] += 1
        except Exception as exc:  # noqa: BLE001
            parse_errors.append({"file": str(txt), "error": repr(exc)})
    angles = {}
    for angle in ANGLES:
        d = angle_dir(angle_root_arg, angle, train_root)
        angles[angle] = {
            "path": str(d),
            "exists": d.is_dir(),
            "images_exists": (d / "images").is_dir(),
            "annfiles_exists": (d / "annfiles").is_dir(),
            "num_images": len(list((d / "images").glob("*"))) if (d / "images").is_dir() else 0,
            "num_annfiles": len(list((d / "annfiles").glob("*.txt"))) if (d / "annfiles").is_dir() else 0,
        }
    return {
        "train_root": str(train_root),
        "train_root_exists": train_root.is_dir(),
        "structure": {
            "images": img_dir.is_dir(),
            "labelTxt": label_dir.is_dir(),
            "annfiles": ann_dir.is_dir(),
            "annotations": (train_root / "annotations").is_dir(),
            "Step6_Format_labels": step6_dir.is_dir(),
        },
        "annotation_source": str(ann_source),
        "num_train_images": len(image_files),
        "num_train_annfiles": len(ann_files),
        "class_instance_counts": cls_counts,
        "parse_errors": parse_errors[:20],
        "angle_root": str(angle_root_arg),
        "angle_root_exists": angle_root_arg.is_dir(),
        "angles": angles,
    }


def audit_generated_configs(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> dict[str, Any]:
    Config = import_config()
    rows = []
    for spec in specs.values():
        row = {
            "model": spec.key,
            "status": spec.status,
            "original_config": spec.original_config_abs or spec.primary_config or "",
            "generated_config": spec.generated_config,
            "blocked_reason": spec.blocked_reason,
            "train_paths": [],
            "train_only_ss_train": False,
            "forbidden_train_path": False,
            "checkpoint_refs": [],
            "pretrain_type": spec.pretrain_type,
            "forbidden_dota_ckpt_used": spec.forbidden_dota_ckpt_used,
            "test_pipeline_contains_tta": False,
        }
        if spec.generated_config:
            try:
                cfg = Config.fromfile(spec.generated_config)
                train_dl = cfg.get("train_dataloader", {})
                paths: list[str] = []
                collect_paths(train_dl, paths)
                row["train_paths"] = paths
                joined = " ".join(paths)
                row["train_only_ss_train"] = str(args.train_root) in joined and not any(
                    bad in joined for bad in ("trainval", "train+val", "angle_sweep_val", "/val", "test")
                )
                row["forbidden_train_path"] = not row["train_only_ss_train"]
                refs: list[str] = []
                collect_checkpoint_refs(cfg.to_dict(), refs)
                row["checkpoint_refs"] = refs
                row["pretrain_type"], row["forbidden_dota_ckpt_used"], _ = classify_pretrain(refs)
                row["test_pipeline_contains_tta"] = config_has_tta(cfg)
                if row["forbidden_train_path"]:
                    spec.status = "FAILED_FOR_LEAKAGE_RISK"
                    spec.blocked_reason = "Train dataloader path audit failed."
                if row["forbidden_dota_ckpt_used"]:
                    spec.status = "FAILED_FOR_LEAKAGE_RISK"
                    spec.blocked_reason = "Forbidden official DOTA checkpoint risk detected."
            except Exception as exc:  # noqa: BLE001
                row["status"] = "BLOCKED"
                row["blocked_reason"] = f"Config audit failed: {exc!r}"
                spec.status = "BLOCKED"
                spec.blocked_reason = row["blocked_reason"]
        rows.append(row)
    return {"models": rows}


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_inventory_audit_md(args: argparse.Namespace, inventory: dict[str, Any], specs: dict[str, ModelSpec], audit: dict[str, Any]) -> None:
    rows = []
    for spec in specs.values():
        rows.append([
            spec.key,
            spec.status,
            spec.original_config_abs or spec.primary_config or "",
            spec.generated_config or "",
            spec.pretrain_type,
            spec.load_from or "",
            spec.forbidden_dota_ckpt_used,
            spec.blocked_reason,
        ])
    angle_rows = [[a, v["exists"], v["images_exists"], v["annfiles_exists"], v["num_images"], v["num_annfiles"], v["path"]] for a, v in inventory["angles"].items()]
    cls_rows = [[k, v] for k, v in inventory["class_instance_counts"].items()]
    lines = [
        "# ss_train Inventory And Leakage Audit",
        "",
        f"- generated_at: `{now()}`",
        f"- train_root: `{inventory['train_root']}`",
        f"- train_root_exists: `{inventory['train_root_exists']}`",
        f"- num_train_images: `{inventory['num_train_images']}`",
        f"- num_train_annfiles: `{inventory['num_train_annfiles']}`",
        f"- annotation_source: `{inventory['annotation_source']}`",
        "",
        "## Train Structure",
        "",
        md_table(["entry", "exists"], [[k, v] for k, v in inventory["structure"].items()]),
        "",
        "## Class Instance Counts",
        "",
        md_table(["class", "instances"], cls_rows),
        "",
        "## Angle Split Check",
        "",
        md_table(["angle", "exists", "images", "annfiles", "num_images", "num_annfiles", "path"], angle_rows),
        "",
        "## Config Inventory",
        "",
        md_table(["model", "status", "original_config", "generated_config", "pretrain_type", "load_from/init_cfg", "forbidden_dota_ckpt_used", "reason"], rows),
        "",
        "## Generated Config Train Path Audit",
        "",
        md_table(
            ["model", "train_only_ss_train", "forbidden_train_path", "checkpoint_refs", "test_pipeline_contains_tta"],
            [[r["model"], r["train_only_ss_train"], r["forbidden_train_path"], "<br>".join(r["checkpoint_refs"]), r["test_pipeline_contains_tta"]] for r in audit["models"]],
        ),
    ]
    write_text(args.result_md_dir / "exp_sstrain0_inventory_and_leakage_audit.md", "\n".join(lines))


def latest_checkpoint(work_dir: Path) -> Path | None:
    direct = work_dir / "latest.pth"
    if direct.exists():
        return direct
    ckpts = sorted(work_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if ckpts:
        return ckpts[-1]
    return None


def best_checkpoint(work_dir: Path) -> Path | None:
    ckpts = sorted(work_dir.glob("best*.pth"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return ckpts[-1] if ckpts else None


def copy_or_link_checkpoint(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            dst = dst.parent / f"{dst.stem}_{run_stamp()}{dst.suffix}"
        else:
            dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def train_command(args: argparse.Namespace, cfg: str, work_dir: Path, gpu_ids: str, resume: bool) -> list[Any]:
    if "," in gpu_ids and args.distributed_launch == "torchrun":
        port = str(args.dist_port_base + (abs(hash(f"{cfg}:{work_dir}:{time.time()}")) % 1000))
        cmd: list[Any] = [
            args.python_bin, "-m", "torch.distributed.run",
            f"--nproc_per_node={len(gpu_ids.split(','))}",
            f"--master_port={port}",
            "tools/train.py", cfg, "--launcher", "pytorch", "--work-dir", str(work_dir),
        ]
    elif "," in gpu_ids and (args.repo_root / "tools/dist_train.sh").exists():
        port = str(args.dist_port_base + (abs(hash(f"{cfg}:{work_dir}:{time.time()}")) % 1000))
        inner = shell_join([
            "bash", "tools/dist_train.sh", cfg, str(len(gpu_ids.split(","))),
            "--work-dir", str(work_dir),
        ])
        cmd = ["bash", "-lc", f"PATH={args.python_bin.parent}:$PATH PORT={port} {inner}"]
    elif (args.repo_root / "tools/train.py").exists():
        cmd = [args.python_bin, "tools/train.py", cfg, "--work-dir", str(work_dir)]
    else:
        cmd = [args.python_bin, "train.py", cfg, "--work-dir", str(work_dir)]
    if resume:
        cmd.append("--resume")
    return cmd


def test_command(args: argparse.Namespace, cfg: str, ckpt: str, work_dir: Path, out_pkl: Path, gpu_ids: str) -> list[Any]:
    if "," in gpu_ids and (args.repo_root / "tools/dist_test.sh").exists():
        return ["bash", "tools/dist_test.sh", cfg, ckpt, str(len(gpu_ids.split(","))), "--work-dir", str(work_dir), "--out", str(out_pkl)]
    return [args.python_bin, "tools/test.py", cfg, ckpt, "--work-dir", str(work_dir), "--out", str(out_pkl)]


def run_smoke(args: argparse.Namespace, specs: dict[str, ModelSpec], logger: CommandLogger) -> dict[str, Any]:
    selected = resolve_models(args, specs)
    if args.only_model:
        selected = [args.only_model]
    elif "retinanet_msrr" in selected:
        selected = ["retinanet_msrr"]
    results = []
    for key in selected:
        spec = specs[key]
        if spec.status not in {"GENERATED", "TRAIN_DONE", "SMOKE_DONE"}:
            results.append({"model": key, "status": spec.status, "reason": spec.blocked_reason})
            continue
        bs = args.batch_size or spec.batch_size_per_gpu or 1
        smoke_cfg = args.work_dir / "generated_configs" / f"{key}_smoke.py"
        smoke_work = args.work_dir / "smoke" / key
        write_text(smoke_cfg, generate_smoke_config_text(Path(spec.generated_config), args.max_iters_for_smoke, args.max_images_for_smoke, bs, 0, smoke_work))
        spec.smoke_config = str(smoke_cfg)
        train_meta = logger.run(
            f"smoke_train_{key}",
            train_command(args, str(smoke_cfg), smoke_work, args.gpu_ids, resume=False),
            gpu_ids=args.gpu_ids,
            timeout=7200,
        )
        ckpt = latest_checkpoint(smoke_work)
        eval_meta = None
        if ckpt and train_meta["return_code"] == 0:
            eval_dir = args.work_dir / "smoke_eval" / key / "angle_000"
            eval_cfg = args.work_dir / "generated_configs" / f"{key}_smoke_eval_angle_000.py"
            write_text(eval_cfg, generate_angle_eval_config_text(
                Path(spec.generated_config),
                angle_dir(args.angle_root, "000", args.train_root),
                "000",
                0,
                eval_dir,
                indices=args.max_images_for_smoke,
            ))
            eval_meta = logger.run(
                f"smoke_eval_{key}_angle000",
                test_command(args, str(eval_cfg), str(ckpt), eval_dir, eval_dir / "predictions.pkl", args.gpu_ids),
                gpu_ids=args.gpu_ids,
                timeout=7200,
            )
        status = "DONE" if train_meta["return_code"] == 0 and eval_meta and eval_meta["return_code"] == 0 else "FAILED"
        spec.status = "SMOKE_DONE" if status == "DONE" else train_meta.get("error_class") or "FAILED"
        results.append({"model": key, "status": status, "train": train_meta, "eval": eval_meta, "checkpoint": str(ckpt) if ckpt else ""})
    payload = {"generated_at": now(), "gpu_ids": args.gpu_ids, "results": results}
    write_json(args.work_dir / "results" / "smoke_test.json", payload)
    lines = ["# ss_train Smoke Test", "", f"- generated_at: `{now()}`", f"- gpu_ids: `{args.gpu_ids}`", ""]
    rows = []
    for r in results:
        train_rc = r.get("train", {}).get("return_code", "NA") if isinstance(r.get("train"), dict) else "NA"
        eval_rc = r.get("eval", {}).get("return_code", "NA") if isinstance(r.get("eval"), dict) else "NA"
        rows.append([r["model"], r["status"], train_rc, eval_rc, r.get("checkpoint", ""), r.get("reason", "")])
    lines.append(md_table(["model", "status", "train_rc", "eval_rc", "checkpoint", "reason"], rows))
    write_text(args.result_md_dir / "exp_sstrain1_smoke_test.md", "\n".join(lines))
    return payload


def run_batchsize_smoke(args: argparse.Namespace, specs: dict[str, ModelSpec], logger: CommandLogger) -> dict[str, Any]:
    existing_path = args.work_dir / "results" / "multigpu_batchsize.json"
    existing_results: list[dict[str, Any]] = []
    if existing_path.exists():
        try:
            existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
            existing_results = list(existing_payload.get("results", []))
        except Exception:
            existing_results = []
    existing_by_model = {r.get("model"): r for r in existing_results if r.get("model")}
    results = [r for r in existing_results if r.get("model") not in set(resolve_models(args, specs))]
    candidates = [int(x) for x in args.batch_size_candidates.split(",") if x.strip()]
    for key in resolve_models(args, specs):
        spec = specs[key]
        if spec.status not in {"GENERATED", "SMOKE_DONE", "TRAIN_DONE"}:
            results.append({"model": key, "status": spec.status, "reason": spec.blocked_reason})
            continue
        successes = []
        failures = []
        for bs in candidates:
            smoke_cfg = args.work_dir / "generated_configs" / f"{key}_bs{bs}_smoke.py"
            smoke_work = args.work_dir / "batchsize_smoke" / key / f"bs{bs}"
            write_text(smoke_cfg, generate_smoke_config_text(Path(spec.generated_config), args.max_iters_for_smoke, args.max_images_for_smoke, bs, args.num_workers, smoke_work))
            meta = logger.run(
                f"batchsize_smoke_{key}_bs{bs}",
                train_command(args, str(smoke_cfg), smoke_work, args.gpu_ids, resume=False),
                gpu_ids=args.gpu_ids,
                timeout=args.smoke_timeout_sec,
            )
            if meta["return_code"] == 0:
                successes.append({"batch_size": bs, "meta": meta})
                spec.batch_size_per_gpu = bs
                spec.global_batch_size = bs * len(args.gpu_ids.split(","))
            else:
                failures.append({"batch_size": bs, "error_class": meta.get("error_class", ""), "meta": meta})
                if meta.get("error_class") == "FAILED_OOM":
                    continue
        results.append({
            "model": key,
            "status": "DONE" if successes else "FAILED",
            "candidates": candidates,
            "successes": [x["batch_size"] for x in successes],
            "failures": [{"batch_size": x["batch_size"], "error_class": x["error_class"]} for x in failures],
            "max_stable_batch_size": max([x["batch_size"] for x in successes], default=None),
            "global_batch_size": spec.global_batch_size,
            "ddp_success": bool(successes) and "," in args.gpu_ids,
        })
    results = sorted(results, key=lambda r: MODEL_ORDER.index(r["model"]) if r.get("model") in MODEL_ORDER else 999)
    payload = {"generated_at": now(), "gpu_ids": args.gpu_ids, "results": results}
    write_json(args.work_dir / "results" / "multigpu_batchsize.json", payload)
    rows = [[r["model"], r["status"], r.get("candidates", ""), r.get("successes", ""), r.get("failures", ""), r.get("max_stable_batch_size", ""), r.get("global_batch_size", ""), r.get("ddp_success", "")] for r in results]
    write_text(
        args.result_md_dir / "exp_sstrain2_multigpu_batchsize.md",
        "# ss_train Multi-GPU Batch Size Smoke\n\n" + md_table(["model", "status", "candidates", "successes", "failures", "max_per_gpu", "global_bs", "ddp_success"], rows),
    )
    return payload


def parse_last_loss(log_text: str) -> tuple[str, bool]:
    loss = ""
    nan = "nan" in log_text.lower()
    for line in reversed(log_text.splitlines()):
        if "loss" in line.lower():
            loss = line.strip()
            break
    return loss, nan


def run_train(args: argparse.Namespace, specs: dict[str, ModelSpec], logger: CommandLogger) -> dict[str, Any]:
    results = []
    for key in MODEL_ORDER:
        if key not in resolve_models(args, specs):
            continue
        spec = specs[key]
        if spec.status in {"BLOCKED", "FAILED_FOR_LEAKAGE_RISK"} or not spec.generated_config:
            results.append({"model": key, "status": spec.status, "reason": spec.blocked_reason})
            write_train_md(args, spec, None)
            continue
        work = args.work_dir / "train" / key
        meta = logger.run(
            f"train_{key}",
            train_command(args, spec.generated_config, work, args.gpu_ids, resume=args.resume),
            gpu_ids=args.gpu_ids,
            timeout=None,
        )
        ckpt = latest_checkpoint(work)
        best = best_checkpoint(work)
        if ckpt:
            spec.latest_checkpoint = str(ckpt)
            copy_or_link_checkpoint(ckpt, args.save_ckpt_dir / key / "latest.pth", args.force)
        if best:
            spec.best_checkpoint = str(best)
        spec.train_log = meta["stdout_path"]
        log_text = read_text(Path(meta["stdout_path"])) + "\n" + read_text(Path(meta["stderr_path"]))
        last_loss, nan = parse_last_loss(log_text)
        spec.final_epoch_or_iter = last_loss[:200]
        spec.status = "TRAIN_DONE" if meta["return_code"] == 0 and ckpt else meta.get("error_class") or "FAILED"
        write_train_md(args, spec, meta, nan=nan)
        results.append({"model": key, "status": spec.status, "meta": meta, "checkpoint": spec.latest_checkpoint})
        if ckpt and meta["return_code"] == 0:
            eval_dir = args.work_dir / "post_train_angle000" / key
            eval_cfg = args.work_dir / "generated_configs" / f"{key}_posttrain_angle_000.py"
            write_text(eval_cfg, generate_angle_eval_config_text(Path(spec.generated_config), angle_dir(args.angle_root, "000", args.train_root), "000", args.num_workers, eval_dir))
            logger.run(
                f"post_train_eval_{key}_angle000",
                test_command(args, str(eval_cfg), str(ckpt), eval_dir, eval_dir / "predictions.pkl", args.gpu_ids),
                gpu_ids=args.gpu_ids,
                timeout=7200,
            )
    payload = {"generated_at": now(), "results": results}
    write_json(args.work_dir / "results" / "training_results.json", payload)
    return payload


def write_train_md(args: argparse.Namespace, spec: ModelSpec, meta: dict[str, Any] | None, nan: bool = False) -> None:
    status = spec.status
    oom = bool(meta and meta.get("error_class") == "FAILED_OOM")
    lines = [
        f"# Train {spec.display} ss_train",
        "",
        f"- model: `{spec.key}`",
        f"- original_config: `{spec.original_config_abs or spec.primary_config or ''}`",
        f"- generated_config: `{spec.generated_config}`",
        f"- train_data_root: `{args.train_root}`",
        "- verified_train_only_ss_train: `True`" if status not in {"BLOCKED", "FAILED_FOR_LEAKAGE_RISK"} else "- verified_train_only_ss_train: `False`",
        f"- load_from/init_cfg: `{spec.load_from}`",
        f"- pretrain_type: `{spec.pretrain_type}`",
        f"- forbidden_dota_official_checkpoint_used: `{spec.forbidden_dota_ckpt_used}`",
        f"- gpu: `{args.gpu_ids}`",
        f"- batch_size_per_gpu: `{spec.batch_size_per_gpu}`",
        f"- global_batch_size: `{spec.global_batch_size}`",
        f"- lr: `{spec.lr}`",
        f"- max_epoch_or_iter: `{spec.final_epoch_or_iter}`",
        f"- optimizer: `{spec.lr}`",
        f"- amp: `{spec.amp}`",
        f"- ms_aug: `{spec.ms_aug}`",
        f"- rr_aug: `{spec.rr_aug}`",
        f"- train_start_time: `{meta.get('start_time') if meta else ''}`",
        f"- train_end_time: `{meta.get('end_time') if meta else ''}`",
        f"- final_checkpoint: `{spec.latest_checkpoint}`",
        f"- best_checkpoint: `{spec.best_checkpoint}`",
        f"- last_loss: `{spec.final_epoch_or_iter}`",
        f"- loss_nan: `{nan}`",
        f"- oom: `{oom}`",
        f"- train_stdout_log: `{meta.get('stdout_path') if meta else ''}`",
        f"- train_stderr_log: `{meta.get('stderr_path') if meta else ''}`",
        f"- status: `{status}`",
        f"- blocked_reason: `{spec.blocked_reason}`",
    ]
    write_text(args.result_md_dir / f"train_{spec.key}_ss_train.md", "\n".join(lines))


def parse_metric_from_text(text: str) -> float | None:
    candidates = []
    for pat in [
        r"'mAP'\s*:\s*([0-9]*\.?[0-9]+)",
        r'"mAP"\s*:\s*([0-9]*\.?[0-9]+)',
        r"\bmAP\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bAP50\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]:
        for m in re.finditer(pat, text):
            try:
                candidates.append(float(m.group(1)))
            except ValueError:
                pass
    return candidates[-1] if candidates else None


def count_predictions(pkl_path: Path) -> tuple[int, float]:
    if not pkl_path.exists():
        return 0, 0.0
    code = (
        "import pickle, sys, json\n"
        "p=sys.argv[1]\n"
        "data=pickle.load(open(p,'rb'))\n"
        "n=0; scores=[]\n"
        "for item in data:\n"
        "    inst=item.get('pred_instances',{}) if isinstance(item,dict) else {}\n"
        "    s=inst.get('scores', []) if isinstance(inst,dict) else []\n"
        "    try:\n"
        "        vals=s.tolist()\n"
        "    except Exception:\n"
        "        vals=list(s) if hasattr(s,'__iter__') else []\n"
        "    n+=len(vals); scores.extend(float(x) for x in vals)\n"
        "print(json.dumps({'num_predictions':n,'mean_score':sum(scores)/len(scores) if scores else 0.0,'num_samples':len(data)}))\n"
    )
    rc, out, _ = quick_run([sys.executable, "-c", code, str(pkl_path)], pkl_path.parent, timeout=120)
    if rc == 0:
        try:
            d = json.loads(out.strip().splitlines()[-1])
            return int(d.get("num_predictions", 0)), float(d.get("mean_score", 0.0))
        except Exception:
            return 0, 0.0
    return 0, 0.0


def num_images_for_angle(angle_path: Path) -> int:
    img = angle_path / "images"
    return len([p for p in img.glob("*") if p.is_file()]) if img.is_dir() else 0


def run_eval(args: argparse.Namespace, specs: dict[str, ModelSpec], logger: CommandLogger) -> dict[str, Any]:
    rows = []
    failed = []
    selected_angles = [args.only_angle] if args.only_angle else ANGLES
    for key in resolve_models(args, specs):
        spec = specs[key]
        ckpt = pick_eval_checkpoint(args, spec)
        model_rows = []
        if not ckpt:
            failed.append({"model": key, "status": "CHECKPOINT_MISSING", "reason": "No latest/final checkpoint found."})
            spec.status = "EVAL_FAILED"
            write_eval_md(args, spec, model_rows, "CHECKPOINT_MISSING")
            continue
        for angle in selected_angles:
            aroot = angle_dir(args.angle_root, angle, args.train_root)
            eval_dir = args.work_dir / "eval" / key / f"angle_{angle}"
            cfg_path = args.work_dir / "generated_configs" / f"{key}_eval_angle_{angle}.py"
            write_text(cfg_path, generate_angle_eval_config_text(Path(spec.generated_config), aroot, angle, args.num_workers, eval_dir))
            out_pkl = eval_dir / "predictions.pkl"
            meta = logger.run(
                f"eval_{key}_angle_{angle}",
                test_command(args, str(cfg_path), str(ckpt), eval_dir, out_pkl, args.gpu_ids),
                gpu_ids=args.gpu_ids,
                timeout=14400,
            )
            text = read_text(Path(meta["stdout_path"])) + "\n" + read_text(Path(meta["stderr_path"]))
            metric = parse_metric_from_text(text)
            n_pred, mean_score = count_predictions(out_pkl)
            no_tta = "--tta" not in meta["command"] and "TestTimeAug" not in read_text(cfg_path)
            row = {
                "model": key,
                "split": "ss_train_retrain",
                "checkpoint_type": "latest_or_final",
                "checkpoint_path": str(ckpt),
                "angle": angle,
                "ap50": metric,
                "map": metric,
                "num_images": num_images_for_angle(aroot),
                "num_predictions": n_pred,
                "mean_score": mean_score,
                "eval_status": "DONE" if meta["return_code"] == 0 and metric is not None else meta.get("error_class") or "EVAL_FAILED",
                "eval_log": meta["stdout_path"],
                "config_path": str(cfg_path),
                "data_root": str(aroot),
                "no_tta_verified": no_tta,
            }
            rows.append(row)
            model_rows.append(row)
            write_json(eval_dir / "metric.json", row)
            if row["eval_status"] != "DONE":
                failed.append({"model": key, "angle": angle, "status": row["eval_status"], "meta": meta})
        write_eval_md(args, spec, model_rows, "")
    payload = {"generated_at": now(), "rows": rows, "failed": failed}
    write_json(args.work_dir / "results" / "eval_results.json", payload)
    write_metrics_outputs(args, specs, rows, failed)
    return payload


def pick_eval_checkpoint(args: argparse.Namespace, spec: ModelSpec) -> Path | None:
    if args.eval_best and spec.best_checkpoint:
        return Path(spec.best_checkpoint)
    candidates = [
        args.save_ckpt_dir / spec.key / "latest.pth",
        Path(spec.latest_checkpoint) if spec.latest_checkpoint else Path(),
        latest_checkpoint(args.work_dir / "train" / spec.key) or Path(),
    ]
    for p in candidates:
        if p and p.exists():
            return p
    return None


def write_eval_md(args: argparse.Namespace, spec: ModelSpec, rows: list[dict[str, Any]], error: str) -> None:
    table_rows = []
    vals = []
    for r in rows:
        v = r.get("map")
        if isinstance(v, (int, float)):
            vals.append(float(v))
        table_rows.append([r.get("angle", ""), fmt(v), r.get("eval_status", ""), r.get("num_images", ""), r.get("num_predictions", ""), r.get("no_tta_verified", ""), r.get("eval_log", "")])
    lines = [
        f"# Eval {spec.display} ss_train 12-angle no-TTA",
        "",
        f"- generated_at: `{now()}`",
        f"- checkpoint: `{rows[0]['checkpoint_path'] if rows else ''}`",
        f"- error: `{error}`",
        f"- mean: `{fmt(sum(vals) / len(vals) if vals else None)}`",
        f"- min: `{fmt(min(vals) if vals else None)}`",
        f"- max: `{fmt(max(vals) if vals else None)}`",
        f"- range: `{fmt((max(vals) - min(vals)) if vals else None)}`",
        f"- std: `{fmt(std(vals) if vals else None)}`",
        f"- RSI: `{fmt((min(vals) / (sum(vals) / len(vals))) if vals and sum(vals) else None)}`",
        "",
        md_table(["angle", "mAP@0.5", "status", "num_images", "num_predictions", "no_tta", "eval_log"], table_rows),
    ]
    write_text(args.result_md_dir / f"eval_{spec.key}_ss_train_12angle_no_tta.md", "\n".join(lines))


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "NA"
    if isinstance(v, (int, float)):
        if math.isnan(float(v)):
            return "NA"
        return f"{float(v):.4f}"
    return str(v)


def std(vals: list[float]) -> float | None:
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def stats_from_values(vals_by_angle: dict[str, float | None]) -> dict[str, float | None]:
    vals = [float(vals_by_angle[a]) for a in ANGLES if isinstance(vals_by_angle.get(a), (int, float))]
    if not vals:
        return {"mean": None, "min": None, "max": None, "range": None, "std": None, "RSI": None}
    mean = sum(vals) / len(vals)
    mn = min(vals)
    mx = max(vals)
    return {"mean": mean, "min": mn, "max": mx, "range": mx - mn, "std": std(vals), "RSI": mn / mean if mean else None}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_metrics_outputs(args: argparse.Namespace, specs: dict[str, ModelSpec], rows: list[dict[str, Any]], failed: list[dict[str, Any]]) -> None:
    results_dir = args.work_dir / "results"
    fields = [
        "model", "split", "checkpoint_type", "checkpoint_path", "angle", "ap50", "map",
        "num_images", "num_predictions", "mean_score", "eval_status", "eval_log",
        "config_path", "data_root", "no_tta_verified",
    ]
    write_csv(results_dir / "sstrain_12angle_metrics.csv", rows, fields)
    write_json(results_dir / "sstrain_12angle_metrics.json", rows)
    by_model = {(r["model"], r["angle"]): r.get("map") for r in rows}
    delta_rows = []
    for model in SUMMARY_ORDER:
        for angle in ANGLES:
            ss = by_model.get((model, angle))
            off = OFFICIAL_BASELINE[model][angle]
            delta_rows.append({"model": model, "angle": angle, "official": off, "sstrain": ss, "delta": (ss - off) if isinstance(ss, (int, float)) else ""})
    write_csv(results_dir / "sstrain_vs_official_delta.csv", delta_rows, ["model", "angle", "official", "sstrain", "delta"])
    audit_rows = []
    for spec in specs.values():
        audit_rows.append({
            "model": spec.key,
            "status": spec.status,
            "original_config": spec.original_config_abs or spec.primary_config or "",
            "generated_config": spec.generated_config,
            "train_root": str(args.train_root),
            "verified_train_only_ss_train": spec.status not in {"FAILED_FOR_LEAKAGE_RISK", "BLOCKED"},
            "load_from": spec.load_from,
            "pretrain_type": spec.pretrain_type,
            "forbidden_dota_ckpt_used": spec.forbidden_dota_ckpt_used,
            "latest_checkpoint": spec.latest_checkpoint or str(args.save_ckpt_dir / spec.key / "latest.pth"),
            "final_epoch_or_iter": spec.final_epoch_or_iter,
            "batch_size_per_gpu": spec.batch_size_per_gpu,
            "global_batch_size": spec.global_batch_size,
            "lr": spec.lr,
            "amp": spec.amp,
            "ms_aug": spec.ms_aug,
            "rr_aug": spec.rr_aug,
            "train_log": spec.train_log,
        })
    write_json(results_dir / "training_audit.json", audit_rows)
    write_json(results_dir / "config_inventory.json", {k: spec.__dict__ for k, spec in specs.items()})
    write_json(results_dir / "failed_jobs.json", failed)
    write_summary_md(args, specs, rows, failed, audit_rows)


def table_for_metric(values: dict[str, dict[str, Any]], include_stats: bool = True) -> str:
    headers = ["angle", *[DISPLAY_NAMES[m] for m in SUMMARY_ORDER]]
    rows = []
    for angle in ANGLES:
        rows.append([angle, *[fmt(values.get(m, {}).get(angle)) for m in SUMMARY_ORDER]])
    if include_stats:
        rows.append(["mean", *[fmt(values.get(m, {}).get("mean")) for m in SUMMARY_ORDER]])
        rows.append(["range", *[fmt(values.get(m, {}).get("range")) for m in SUMMARY_ORDER]])
    return md_table(headers, rows)


def write_summary_md(args: argparse.Namespace, specs: dict[str, ModelSpec], rows: list[dict[str, Any]], failed: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    by_model_values: dict[str, dict[str, Any]] = {m: {} for m in SUMMARY_ORDER}
    for r in rows:
        if isinstance(r.get("map"), (int, float)):
            by_model_values[r["model"]][r["angle"]] = float(r["map"])
    for model in SUMMARY_ORDER:
        st = stats_from_values(by_model_values[model])
        by_model_values[model].update(st)
    delta_values: dict[str, dict[str, Any]] = {m: {} for m in SUMMARY_ORDER}
    for model in SUMMARY_ORDER:
        for angle in ANGLES:
            ss = by_model_values[model].get(angle)
            delta_values[model][angle] = (ss - OFFICIAL_BASELINE[model][angle]) if isinstance(ss, (int, float)) else None
        if isinstance(by_model_values[model].get("mean"), (int, float)):
            delta_values[model]["mean"] = by_model_values[model]["mean"] - OFFICIAL_BASELINE[model]["mean"]
        if isinstance(by_model_values[model].get("range"), (int, float)):
            delta_values[model]["range"] = by_model_values[model]["range"] - OFFICIAL_BASELINE[model]["range"]
    comp_rows = []
    for model in SUMMARY_ORDER:
        off_vals = {a: OFFICIAL_BASELINE[model][a] for a in ANGLES}
        off_stats = stats_from_values(off_vals)
        ss_stats = stats_from_values(by_model_values[model])
        comp_rows.append([
            DISPLAY_NAMES[model],
            fmt(OFFICIAL_BASELINE[model]["mean"]),
            fmt(ss_stats["mean"]),
            fmt(ss_stats["mean"] - OFFICIAL_BASELINE[model]["mean"] if isinstance(ss_stats["mean"], (int, float)) else None),
            fmt(off_stats["min"]),
            fmt(ss_stats["min"]),
            fmt(ss_stats["min"] - off_stats["min"] if isinstance(ss_stats["min"], (int, float)) else None),
            fmt(OFFICIAL_BASELINE[model]["range"]),
            fmt(ss_stats["range"]),
            fmt(ss_stats["range"] - OFFICIAL_BASELINE[model]["range"] if isinstance(ss_stats["range"], (int, float)) else None),
            fmt(off_stats["std"]),
            fmt(ss_stats["std"]),
            fmt(ss_stats["std"] - off_stats["std"] if isinstance(ss_stats["std"], (int, float)) and isinstance(off_stats["std"], (int, float)) else None),
            fmt(off_stats["RSI"]),
            fmt(ss_stats["RSI"]),
            fmt(ss_stats["RSI"] - off_stats["RSI"] if isinstance(ss_stats["RSI"], (int, float)) and isinstance(off_stats["RSI"], (int, float)) else None),
        ])
    audit_table_rows = [[
        DISPLAY_NAMES.get(r["model"], r["model"]), r["status"], r["original_config"], r["generated_config"],
        r["train_root"], r["verified_train_only_ss_train"], r["load_from"], r["pretrain_type"],
        r["forbidden_dota_ckpt_used"], r["latest_checkpoint"], r["final_epoch_or_iter"],
        r["batch_size_per_gpu"], r["global_batch_size"], r["lr"], r["amp"], r["ms_aug"], r["rr_aug"], r["train_log"],
    ] for r in audit_rows]
    success = [DISPLAY_NAMES[k] for k, s in specs.items() if s.status in {"TRAIN_DONE", "EVAL_DONE"}]
    smoke_only = [DISPLAY_NAMES[k] for k, s in specs.items() if s.status == "SMOKE_DONE"]
    config_missing = [DISPLAY_NAMES[k] for k, s in specs.items() if s.status == "BLOCKED"]
    leakage = [DISPLAY_NAMES[k] for k, s in specs.items() if s.status == "FAILED_FOR_LEAKAGE_RISK"]
    oom = [DISPLAY_NAMES[k] for k, s in specs.items() if "OOM" in s.status]
    eval_failed = sorted({DISPLAY_NAMES.get(f.get("model", ""), f.get("model", "")) for f in failed})
    all_no_dota = all(not s.forbidden_dota_ckpt_used for s in specs.values() if s.generated_config)
    lines = [
        "# ss_train Retrain 12-angle No-TTA Summary",
        "",
        f"- generated_at: `{now()}`",
        f"- train_root: `{args.train_root}`",
        f"- checkpoint_policy: `latest/final for main table; no 12-angle best selection`",
        "",
        "## Table 1. Official train+val baseline",
        "",
        table_for_metric(OFFICIAL_BASELINE),
        "",
        "## Table 2. ss_train retrained 12-angle mAP",
        "",
        table_for_metric(by_model_values),
        "",
        "## Table 3. ss_train - official_trainval",
        "",
        table_for_metric(delta_values),
        "",
        "## Table 4. Stability Comparison",
        "",
        md_table([
            "model", "official_mean", "sstrain_mean", "delta_mean", "official_min", "sstrain_min",
            "delta_min", "official_range", "sstrain_range", "delta_range", "official_std",
            "sstrain_std", "delta_std", "official_RSI", "sstrain_RSI", "delta_RSI",
        ], comp_rows),
        "",
        "## Table 5. Training Audit",
        "",
        md_table([
            "model", "status", "original_config", "generated_config", "train_root",
            "verified_train_only_ss_train", "load_from", "pretrain_type",
            "forbidden_dota_ckpt_used", "latest_checkpoint", "final_epoch_or_iter",
            "batch_size_per_gpu", "global_batch_size", "lr", "amp", "ms_aug",
            "rr_aug", "train_log",
        ], audit_table_rows),
        "",
        "## Table 6. Failures And Risks",
        "",
        f"- successful_training: `{success}`",
        f"- smoke_only: `{smoke_only}`",
        f"- config_missing_or_blocked: `{config_missing}`",
        f"- leakage_blocked: `{leakage}`",
        f"- oom_failed: `{oom}`",
        f"- eval_failed: `{eval_failed}`",
        f"- all_ss_train_weights_without_official_dota_trainval_checkpoint: `{all_no_dota}`",
        "- main_table_uses_latest_or_final_not_angle_selected_best: `True`",
    ]
    write_text(args.result_md_dir / "summary_sstrain_retrain_12angle_no_tta.md", "\n".join(lines))


def resolve_models(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> list[str]:
    if args.only_model:
        return [args.only_model]
    if args.models == "all":
        return [m for m in MODEL_ORDER if m in specs]
    selected = []
    for item in args.models.split(","):
        item = item.strip()
        if item:
            selected.append(item)
    return selected


def setup_and_audit(args: argparse.Namespace) -> dict[str, ModelSpec]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "results").mkdir(parents=True, exist_ok=True)
    specs = find_configs(args.repo_root)
    generate_configs(args, specs)
    inventory = dataset_inventory(args.train_root, args.angle_root)
    audit = audit_generated_configs(args, specs)
    write_json(args.work_dir / "results" / "config_inventory.json", {k: s.__dict__ for k, s in specs.items()})
    write_json(args.work_dir / "results" / "inventory_and_leakage_audit.json", {"inventory": inventory, "audit": audit})
    write_inventory_audit_md(args, inventory, specs, audit)
    return specs


def write_final_machine_files_if_missing(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> None:
    results_dir = args.work_dir / "results"
    if not (results_dir / "sstrain_12angle_metrics.csv").exists():
        write_metrics_outputs(args, specs, [], [{"status": "NO_EVAL_ROWS"}])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--train-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train"))
    parser.add_argument("--angle-root", type=Path, default=Path("/data1/zcy/OpenRSD/data/DOTA1_1024_500/angle_sweep_val/realistic"))
    parser.add_argument("--work-dir", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/ss_train_retrain_12angle_no_tta"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--save-ckpt-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results/ss_train_retrain"))
    parser.add_argument("--gpu-ids", default="6,7")
    parser.add_argument("--mode", choices=["dryrun", "smoke", "train", "eval", "full", "summary"], default="dryrun")
    parser.add_argument("--models", default="all")
    parser.add_argument("--only-model", choices=MODEL_ORDER)
    parser.add_argument("--only-angle", choices=ANGLES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-size-candidates", default="1,2,4,8")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-iters-for-smoke", type=int, default=20)
    parser.add_argument("--max-images-for-smoke", type=int, default=32)
    parser.add_argument("--smoke-timeout-sec", type=int, default=300)
    parser.add_argument("--dist-port-base", type=int, default=29600)
    parser.add_argument("--distributed-launch", choices=["dist_train", "torchrun"], default="dist_train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-latest", action="store_true")
    parser.add_argument("--eval-best", action="store_true")
    parser.add_argument("--no-tta", action="store_true", default=True)
    default_python = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--python-bin", type=Path, default=default_python if default_python.exists() else Path(sys.executable))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.train_root = args.train_root.resolve()
    args.angle_root = args.angle_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.save_ckpt_dir = args.save_ckpt_dir.resolve()
    args.python_bin = args.python_bin.resolve()
    specs = setup_and_audit(args)
    logger = CommandLogger(args)
    if args.mode == "dryrun":
        write_final_machine_files_if_missing(args, specs)
        return
    if args.mode == "smoke":
        if args.batch_size_candidates and args.gpu_ids == "6,7" and not args.only_model:
            run_batchsize_smoke(args, specs, logger)
        else:
            run_smoke(args, specs, logger)
        write_final_machine_files_if_missing(args, specs)
        return
    if args.mode == "train":
        run_train(args, specs, logger)
        write_final_machine_files_if_missing(args, specs)
        return
    if args.mode == "eval":
        run_eval(args, specs, logger)
        return
    if args.mode == "full":
        smoke = run_smoke(args, specs, logger)
        if any(r.get("status") == "DONE" for r in smoke.get("results", [])):
            run_batchsize_smoke(args, specs, logger)
            run_train(args, specs, logger)
            run_eval(args, specs, logger)
        else:
            write_final_machine_files_if_missing(args, specs)
        return
    if args.mode == "summary":
        write_final_machine_files_if_missing(args, specs)


if __name__ == "__main__":
    main()

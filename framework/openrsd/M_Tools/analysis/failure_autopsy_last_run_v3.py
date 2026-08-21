#!/usr/bin/env python3
"""Failure autopsy and GPU preflight for full-live v3."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from strict_gpu_runner_v3 import nvidia_smi_text, query_gpu_rows, read_text, run_quiet, write_json, write_text


ANGLES12 = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]


def classify_training_error(text: str) -> str:
    low = text.lower()
    checks = [
        ("config import error", ["importerror", "modulenotfounderror", "custom_imports"]),
        ("dataset path error", ["data_root", "dataset", "not found"]),
        ("ann_file error", ["ann_file", "annotation"]),
        ("image path error", ["img_path", "image path", "loadimage"]),
        ("class number mismatch", ["class number", "num_classes", "classes"]),
        ("pipeline transform error", ["pipeline", "transform", "convertboxtype"]),
        ("init_cfg/pretrained error", ["init_cfg", "pretrained", "load checkpoint"]),
        ("optimizer/lr_scheduler error", ["optimizer", "param_scheduler", "lr_scheduler"]),
        ("CUDA/OOM", ["cuda out of memory", "outofmemoryerror", "cuda error"]),
        ("DDP launcher error", ["torch.distributed", "launcher", "local_rank"]),
        ("missing package", ["no module named", "modulenotfounderror"]),
        ("dataset metainfo mismatch", ["metainfo", "category", "airport", "class mismatch"]),
    ]
    for label, needles in checks:
        if any(n in low for n in needles):
            return label
    if "traceback" in low:
        return "TRACEBACK_UNCLASSIFIED"
    return "NO_TRACEBACK_FOUND"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root
    result = args.result_md_dir
    weights = args.weights_dir
    last_stderr = repo / "work_dirs/openrsd_next_priority_gpu45_20260509_200920/P5_logs/P5_baseline_short_ft_train/stderr.log"
    stderr_tail = "\n".join(read_text(last_stderr).splitlines()[-300:])
    checks = {
        "repo_root_exists": repo.exists(),
        "resultmd_writable": result.exists() and result.is_dir(),
        "weights_dir_exists": weights.exists(),
        "gpu_rows": query_gpu_rows(args.gpu_ids),
        "config_exists": (repo / "M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py").exists(),
        "checkpoint_exists": (weights / "MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth").exists(),
        "angle_sweep": {a: (repo / f"data/DOTA1_1024_500/angle_sweep_val/realistic/angle_{a}").exists() for a in ANGLES12},
        "ss_train_exists": (repo / "data/DOTA1_1024_500/ss_train").exists(),
        "ss_train_images_exists": (repo / "data/DOTA1_1024_500/ss_train/images").exists(),
        "ss_train_annfiles_exists": (repo / "data/DOTA1_1024_500/ss_train/annfiles").exists(),
        "tools_train_exists": (repo / "tools/train.py").exists(),
        "tools_test_exists": (repo / "tools/test.py").exists(),
        "last_training_stderr_exists": last_stderr.exists(),
        "last_training_error_class": classify_training_error(stderr_tail),
        "last_training_stderr_tail_300": stderr_tail,
        "nvidia_smi": nvidia_smi_text(args.gpu_ids),
    }
    rc, git, _ = run_quiet(["rtk", "git", "rev-parse", "HEAD"], cwd=repo)
    checks["git_commit"] = git.strip() if rc == 0 else "NA"
    checks["planned_full_gpu_commands"] = [
        "G1: tools/train.py fixed G1 config on CUDA_VISIBLE_DEVICES=4,5 or split GPU4/GPU5",
        "G2: live_full12_openrsd_inference_v3.py --angles all12 --prompt-families F3_orientation_aware,F1_aerial_context,F0_raw_class --heads alignment,fusion",
        "G3: live_full12x12_ovd_tta_v3.py using only G2 pkl mtimes newer than RUN_TS",
        "G4: large_scale_live_hooks_v3.py --max-live-images 512 --angles all12 --heads alignment,fusion",
        "G5: zero_ap_gpu_reactivation_v3.py prompt sweep/support/calibration probes",
    ]
    return checks


def write_md(args: argparse.Namespace, report: dict[str, Any]) -> None:
    lines = [
        "# G0 Failure Autopsy And GPU Preflight",
        "",
        f"- RUN_TS: `{args.run_ts}`",
        f"- work_dir: `{args.work_dir}`",
        f"- gpu_ids: `{args.gpu_ids}`",
        "",
        "## Required Failure Autopsy",
        "",
        "1. 上次 P1 标题像是 full-12，但实际 md 中 source_angles 只有 `['000','030','060','090','120','150']`，target_angles 也只有 `['000','030','060','090','120','150']`，所以它不是 full 12-angle，更不是 12x12 TTA。",
        "2. 上次 P1 主要依赖已有 OVD suite、旧 `exp_ovd4_results.json`、旧 `predictions.pkl` / `merged_predictions.pkl`，不能算新 live GPU inference。",
        "3. 上次 P2 虽然 hook 成功，但命令里 `max-images=16`、`max-live-images=16`，row_count 只有 364，只能算 smoke hook，不是大规模 GPU 诊断。",
        "4. 上次 P3 是 zero-AP drilldown，source_predictions 是旧 `predictions.pkl`，主要是离线统计，不是 GPU-heavy 实验。",
        "5. 上次 P4 的 Definition B Prediction-GT joint-bin AP 仍然是 `NOT_AVAILABLE`，而且主要是离线 per-bin AP，不是 GPU-heavy 实验。",
        "6. 上次 P5 找到 `usable_train_root=/data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train`，但真实 `tools/train.py` 命令 rc=1，训练没有跑起来。",
        "",
        "## DONE_LIVE_GPU Standard",
        "",
        "- 有 live CUDA 进程；",
        "- 有 nvidia-smi before/during/after；",
        "- 有 peak VRAM；",
        "- 有新生成 predictions.pkl / checkpoint / hook csv；",
        "- 新文件 mtime 晚于 RUN_TS；",
        "- live inference 覆盖全量 12 angles；",
        "- live hook 至少 512 images/angle；",
        "- 训练任务必须进入 train loop 或明确修复失败原因并 fallback。",
        "",
        "Any task that only reads old pkl/json/csv is `OFFLINE_ONLY`; any task with no CUDA process or peak VRAM < 2GB is `GPU_NOT_USED`.",
        "",
        "## Static Checks",
        "",
        "| check | value |",
        "|---|---|",
    ]
    for key, value in report.items():
        if key in {"last_training_stderr_tail_300", "nvidia_smi", "planned_full_gpu_commands"}:
            continue
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)[:500]}` |")
    lines.extend(["", "## Planned Full GPU Commands", ""])
    for cmd in report.get("planned_full_gpu_commands", []):
        lines.append(f"- {cmd}")
    lines.extend(["", "## Last Training stderr Tail", "", "```text", report.get("last_training_stderr_tail_300", ""), "```"])
    write_text(args.out_md, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = collect(args)
    write_json(args.out_json, report)
    write_md(args, report)


if __name__ == "__main__":
    main()


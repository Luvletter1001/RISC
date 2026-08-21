#!/usr/bin/env python3
"""Evaluate or collect DOTA2 val metrics for FOCUS-TAC variants."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


OPENRSD_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
EVAL_VARIANTS = (
    "TAC_V00_FOCUS_ep24_eval",
    "TAC_V01_FOCUS_TAC_zero",
    "TAC_V03_FOCUS_TAC_alpha_only_monitored",
    "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
)
KEY_CLASSES = (
    "small-vehicle",
    "large-vehicle",
    "ship",
    "bridge",
    "helipad",
)
VAL_RUNNER = (
    "experiments/rotation_semantic_attractor/scripts/focus_tac_val_runner.py")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_variants(value: str | None) -> list[str]:
    if not value:
        return list(EVAL_VARIANTS)
    return [item.strip() for item in value.split(",") if item.strip()]


def latest_checkpoint(work_dir: Path) -> Path | None:
    last = work_dir / "last_checkpoint"
    if last.exists():
        value = last.read_text(encoding="utf-8").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = work_dir / path
            if path.exists():
                return path
    checkpoints = sorted(
        work_dir.glob("epoch_*.pth"),
        key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def parse_log_metrics(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(
        r"dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+).*?"
        r"dota/IoU_50_Detail:\s*(\{.*\})",
        text))
    if not matches:
        return None
    match = matches[-1]
    detail_text = match.group(3)
    end_marker = "  data_time:"
    if end_marker in detail_text:
        detail_text = detail_text.split(end_marker, 1)[0]
    detail = ast.literal_eval(detail_text)
    return {
        "mAP": float(match.group(1)),
        "AP50": float(match.group(2)),
        "detail": detail,
    }


def make_val_protocol_config(repo_root: Path, config: Path,
                             out_dir: Path) -> Path:
    """Write a merged runtime config whose test loop uses FOCUS val protocol."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    tools_dir = repo_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from openrsd_env import preload_installed_mmengine

    preload_installed_mmengine()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mmengine.config import Config

    cfg = Config.fromfile(str(config))
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    runtime_config = out_dir / "eval_config_val_protocol.py"
    cfg.dump(str(runtime_config))
    return runtime_config


def row_from_metrics(variant_id: str, metrics: dict[str, Any],
                     metric_source: str, checkpoint: Path,
                     log_path: Path) -> dict[str, Any]:
    detail = metrics.get("detail", {})
    row: dict[str, Any] = {
        "variant_id": variant_id,
        "eval_status": "PASS",
        "metric_source": metric_source,
        "checkpoint": str(checkpoint),
        "eval_log": str(log_path),
        "mAP": metrics.get("mAP", ""),
        "AP50": metrics.get("AP50", ""),
    }
    total_dets = 0
    for class_name, item in detail.items():
        total_dets += int(item.get("num_dets", 0))
    row["total_dets"] = total_dets
    row["det_per_img"] = ""
    for class_name in KEY_CLASSES:
        item = detail.get(class_name, {})
        prefix = class_name.replace("-", "_")
        row[f"{prefix}_AP"] = item.get("ap", "")
        row[f"{prefix}_recall"] = item.get("recall", "")
        row[f"{prefix}_dets"] = item.get("num_dets", "")
        row[f"{prefix}_gts"] = item.get("num_gts", "")
    return row


def run_eval(repo_root: Path, variant_id: str, config: Path,
             checkpoint: Path, out_dir: Path, gpu: str,
             loop: str = "val", launcher: str = "none",
             master_port: int = 29531,
             seed: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    pred_path = out_dir / "predictions.pkl"
    runtime_config = make_val_protocol_config(repo_root, config, out_dir)
    env = dict(os.environ)
    pythonpath = str(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
        "MPLCONFIGDIR": "/tmp",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
    })
    nproc_per_node = len([item for item in gpu.split(",") if item.strip()])
    if loop == "test":
        base_cmd = [
            "tools/test_rotate.py",
            str(runtime_config),
            str(checkpoint),
            "--work-dir",
            str(out_dir / "work_dir"),
            "--out",
            str(pred_path),
        ]
        if launcher == "pytorch":
            base_cmd += ["--launcher", "pytorch"]
            cmd = [
                "rtk",
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu}",
                "PYTHONNOUSERSITE=1",
                f"PYTHONPATH={pythonpath}",
                "MPLCONFIGDIR=/tmp",
                "NCCL_ASYNC_ERROR_HANDLING=1",
                "NCCL_IB_DISABLE=1",
                "NCCL_P2P_DISABLE=1",
                "NCCL_DEBUG=WARN",
                str(OPENRSD_PYTHON),
                "-m",
                "torch.distributed.launch",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}",
                *base_cmd,
            ]
        else:
            cmd = [
                "rtk",
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu}",
                "PYTHONNOUSERSITE=1",
                f"PYTHONPATH={pythonpath}",
                "MPLCONFIGDIR=/tmp",
                str(OPENRSD_PYTHON),
                *base_cmd,
            ]
    elif loop == "val":
        base_cmd = [
            VAL_RUNNER,
            str(runtime_config),
            str(checkpoint),
            "--work-dir",
            str(out_dir / "work_dir"),
            "--launcher",
            launcher,
        ]
        if seed is not None:
            base_cmd += ["--seed", str(seed)]
        if launcher == "pytorch":
            cmd = [
                "rtk",
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu}",
                "PYTHONNOUSERSITE=1",
                f"PYTHONPATH={pythonpath}",
                "MPLCONFIGDIR=/tmp",
                "NCCL_ASYNC_ERROR_HANDLING=1",
                "NCCL_IB_DISABLE=1",
                "NCCL_P2P_DISABLE=1",
                "NCCL_DEBUG=WARN",
                str(OPENRSD_PYTHON),
                "-m",
                "torch.distributed.launch",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}",
                *base_cmd,
            ]
        else:
            cmd = [
                "rtk",
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu}",
                "PYTHONNOUSERSITE=1",
                f"PYTHONPATH={pythonpath}",
                "MPLCONFIGDIR=/tmp",
                str(OPENRSD_PYTHON),
                *base_cmd,
            ]
    else:
        raise ValueError(f"Unsupported eval loop: {loop}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Eval failed for {variant_id}; see {log_path}")
    return log_path


def config_and_checkpoint_for_variant(
        repo_root: Path,
        row: dict[str, str],
        train_dir: Path) -> tuple[Path, Path]:
    config = resolve_path(repo_root, row["config_py"])
    payload = read_json(resolve_path(repo_root, row["config_json"]), {})
    focus_checkpoint = Path(payload.get("focus_checkpoint", ""))
    if not focus_checkpoint.is_absolute():
        focus_checkpoint = repo_root / focus_checkpoint
    if row.get("train_required") == "true":
        train_summary = read_json(
            train_dir / "focus_tac_train_summary.json", {})
        for item in train_summary.get("variants", []):
            if item.get("variant_id") == row["variant_id"]:
                checkpoint_value = str(item.get("checkpoint", ""))
                if not checkpoint_value:
                    continue
                checkpoint = Path(checkpoint_value)
                if not checkpoint.is_absolute():
                    checkpoint = repo_root / checkpoint
                if checkpoint.exists():
                    return config, checkpoint
        checkpoint = latest_checkpoint(
            train_dir / row["variant_id"] / "work_dir")
        if checkpoint is not None:
            return config, checkpoint
    return config, focus_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-index", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default=",".join(EVAL_VARIANTS))
    parser.add_argument("--gpu", default="6")
    parser.add_argument(
        "--launcher",
        choices=("none", "pytorch"),
        default="none",
        help="Use pytorch for distributed val/test evaluation.")
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260610,
        help="Fixed eval seed for support prompt sampling and zero checks.")
    parser.add_argument(
        "--loop",
        choices=("val", "test"),
        default="val",
        help="Use val loop by default to match FOCUS-OVD training logs.")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Ignore existing logs and rerun evaluation.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    config_index = args.config_index
    if not config_index.is_absolute():
        config_index = repo_root / config_index
    train_dir = args.train_dir
    if not train_dir.is_absolute():
        train_dir = repo_root / train_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    rows = read_csv(config_index)
    by_variant = {row["variant_id"]: row for row in rows}
    variants = parse_variants(args.variants)
    unknown = sorted(set(variants) - set(by_variant))
    if unknown:
        raise SystemExit(f"Unknown TAC variants: {unknown}")
    eval_rows: list[dict[str, Any]] = []
    for variant_id in variants:
        row = by_variant[variant_id]
        config, checkpoint = config_and_checkpoint_for_variant(
            repo_root, row, train_dir)
        out_dir = output_dir / variant_id
        train_work_dir = train_dir / variant_id / "work_dir"
        discovered_logs = sorted(
            train_work_dir.glob("**/*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True)
        existing_logs: list[Path] = []
        for candidate in [
                out_dir / "eval.log",
                train_dir / variant_id / "logs" / "train.log",
                *discovered_logs]:
            if candidate not in existing_logs:
                existing_logs.append(candidate)
        metrics = None
        source_log = out_dir / "eval.log"
        if not args.force_run:
            for candidate in existing_logs:
                parsed = parse_log_metrics(candidate)
                if parsed is not None:
                    metrics = parsed
                    source_log = candidate
                    break
        if metrics is None and not args.skip_run:
            source_log = run_eval(
                repo_root,
                variant_id,
                config,
                checkpoint,
                out_dir,
                args.gpu,
                loop=args.loop,
                launcher=args.launcher,
                master_port=args.master_port,
                seed=args.seed)
            metrics = parse_log_metrics(source_log)
        if metrics is None:
            eval_rows.append({
                "variant_id": variant_id,
                "eval_status": "MISSING_METRICS",
                "checkpoint": str(checkpoint),
                "eval_log": str(source_log),
            })
            continue
        eval_rows.append(row_from_metrics(
            variant_id,
            metrics,
            "parsed_log" if source_log.exists() else "missing",
            checkpoint,
            source_log))

    write_csv(output_dir / "focus_tac_eval_summary.csv", eval_rows)
    write_json(output_dir / "focus_tac_eval_summary.json", {
        "status": (
            "PASS_FOCUS_TAC_EVAL"
            if all(row.get("eval_status") == "PASS" for row in eval_rows)
            else "PARTIAL_FOCUS_TAC_EVAL"),
        "rows": eval_rows,
    })
    print(json.dumps({
        "status": (
            "PASS_FOCUS_TAC_EVAL"
            if all(row.get("eval_status") == "PASS" for row in eval_rows)
            else "PARTIAL_FOCUS_TAC_EVAL"),
        "rows": eval_rows,
    }, indent=2, ensure_ascii=False))
    return 0 if all(row.get("eval_status") == "PASS" for row in eval_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

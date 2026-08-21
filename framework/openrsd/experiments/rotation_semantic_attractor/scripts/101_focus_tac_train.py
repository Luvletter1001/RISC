#!/usr/bin/env python3
"""Train selected FOCUS-TAC variants from the generated config index."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


OPENRSD_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
DEFAULT_VARIANTS = (
    "TAC_V03_FOCUS_TAC_alpha_only_monitored",
    "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        return list(DEFAULT_VARIANTS)
    return [item.strip() for item in value.split(",") if item.strip()]


def safety_scan(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    forbidden = [
        "use_declip_support=True",
        "use_declip_support = True",
        "support_type='visual'",
        "support_type = 'visual'",
        "with_aux_bbox_head=False",
        "with_aux_bbox_head = False",
        "dual_fusion=dict",
        "eqtext=dict",
    ]
    hits = [needle for needle in forbidden if needle in text]
    if hits:
        raise RuntimeError(
            f"FOCUS-TAC safety scan failed for {config_path}: {hits}")


def make_runtime_config(src: Path, dst: Path,
                        max_epochs: int, seed: int,
                        val_interval: int) -> Path:
    text = src.read_text(encoding="utf-8")
    text += "\n"
    text += f"max_epochs = {int(max_epochs)}\n"
    text += (
        "train_cfg = dict(type='EpochBasedTrainLoop', "
        f"max_epochs=max_epochs, val_interval={int(val_interval)})\n")
    text += f"randomness = dict(seed={int(seed)}, deterministic=False)\n"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst


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


def train_one(repo_root: Path, row: dict[str, str], output_dir: Path,
              max_epochs: int, seed: int, port: int,
              val_interval: int, gpu: str) -> dict[str, Any]:
    variant_id = row["variant_id"]
    config_src = Path(row["config_py"])
    if not config_src.is_absolute():
        config_src = repo_root / config_src
    safety_scan(config_src)
    variant_dir = output_dir / variant_id
    work_dir = variant_dir / "work_dir"
    log_path = variant_dir / "logs" / "train.log"
    runtime_config = make_runtime_config(
        config_src,
        variant_dir / "config.py",
        max_epochs=max_epochs,
        seed=seed,
        val_interval=val_interval)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "PYTHONNOUSERSITE=1",
        "PORT=" + str(port),
        "PYTHON=" + str(OPENRSD_PYTHON),
        "bash",
        "tools/my_dist_train.sh",
        str(runtime_config),
        "2",
        "--work-dir",
        str(work_dir),
    ]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTHONNOUSERSITE": "1",
        "PORT": str(port),
        "PYTHON": str(OPENRSD_PYTHON),
    })
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            check=False)
    checkpoint = latest_checkpoint(work_dir)
    audit_path = variant_dir / "trainable_audit.json"
    return {
        "variant_id": variant_id,
        "returncode": proc.returncode,
        "status": "PASS_TRAIN" if proc.returncode == 0 and checkpoint else "FAIL_TRAIN",
        "config": str(runtime_config),
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint or ""),
        "train_log": str(log_path),
        "trainable_audit": str(audit_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--base-port", type=int, default=29710)
    parser.add_argument(
        "--gpu",
        default="6,9",
        help="CUDA_VISIBLE_DEVICES for two-card TAC training.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    config_index = args.config_index
    if not config_index.is_absolute():
        config_index = repo_root / config_index
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    requested = set(parse_variants(args.variants))
    rows = read_csv(config_index)
    by_variant = {row["variant_id"]: row for row in rows}
    unknown = sorted(requested - set(by_variant))
    if unknown:
        raise SystemExit(f"Unknown TAC variants: {unknown}")

    results: list[dict[str, Any]] = []
    for idx, variant_id in enumerate([v for v in by_variant if v in requested]):
        row = by_variant[variant_id]
        if row.get("train_required") != "true":
            results.append({
                "variant_id": variant_id,
                "status": "SKIP_EVAL_ONLY",
                "returncode": 0,
            })
            continue
        results.append(train_one(
            repo_root,
            row,
            output_dir,
            max_epochs=args.max_epochs,
            seed=args.seed,
            port=args.base_port + idx,
            val_interval=args.val_interval,
            gpu=args.gpu))

    summary = {
        "status": (
            "PASS_FOCUS_TAC_TRAIN"
            if all(row.get("returncode") == 0 for row in results)
            else "FAIL_FOCUS_TAC_TRAIN"),
        "variants": results,
    }
    write_csv(output_dir / "focus_tac_train_summary.csv", results)
    write_json(output_dir / "focus_tac_train_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

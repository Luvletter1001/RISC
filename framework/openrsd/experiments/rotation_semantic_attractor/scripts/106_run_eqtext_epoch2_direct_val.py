#!/usr/bin/env python3
"""Run strict EQText variants to epoch2 direct validation with kill rules."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_eqtext_strict_epoch2_audit_20260610")
OPENRSD_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
FOCUS_EPOCH2_SV_DETS = 347_552
FAIL_MAP_MIN = 0.62
FAIL_SV_AP_MIN = 0.50
FAIL_SV_DETS_MULT = 1.5
GEOMETRY_COS_MAX = 0.90
GEOMETRY_DELTA_MAX = 0.05


def resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safety_scan(config_path: Path) -> list[str]:
    text = config_path.read_text(encoding="utf-8")
    failures: list[str] = []
    forbidden_literals = [
        "support_type='visual'",
        'support_type="visual"',
        "support_type = 'visual'",
        'support_type = "visual"',
        "use_declip_support=True",
        "use_declip_support = True",
        "with_aux_bbox_head=False",
        "with_aux_bbox_head = False",
        "DOTA1",
        "angle_sweep_val",
    ]
    failures.extend([item for item in forbidden_literals if item in text])
    max_epochs_match = re.findall(r"max_epochs\s*=\s*(\d+)", text)
    if not max_epochs_match or int(max_epochs_match[-1]) > 2:
        failures.append("max_epochs_missing_or_gt_2")
    val_interval_match = re.findall(r"val_interval\s*=\s*(\d+)", text)
    if not val_interval_match or int(val_interval_match[-1]) != 2:
        failures.append("val_interval_not_2")
    anti_matches = re.findall(
        r"anti_attractor_weight['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text)
    for value in anti_matches:
        if float(value) > 0.0:
            failures.append("anti_attractor_weight_gt_0")
            break
    if "with_aux_bbox_head=True" not in text and "with_aux_bbox_head = True" not in text:
        failures.append("with_aux_bbox_head_true_not_explicit")
    if "support_type='text'" not in text and 'support_type="text"' not in text:
        failures.append("support_type_text_not_explicit")
    return failures


def run_command(repo_root: Path, cmd: list[str], log_path: Path,
                timeout_s: int) -> int:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "6,9",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
                check=False)
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def latest_log(work_dir: Path, fallback: Path) -> Path:
    logs = sorted(
        list(work_dir.glob("*.log")) + list(work_dir.glob("*/*.log")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return logs[-1] if logs else fallback


def parse_metrics_from_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, Any] = {}
    matches = list(re.finditer(
        r"Epoch\(val\) \[2\]\[\s*\d+/\d+\].*?dota/mAP:\s*"
        r"(?P<map>[-+]?\d+(?:\.\d+)?)\s+dota/AP50:\s*"
        r"(?P<ap50>[-+]?\d+(?:\.\d+)?).*?"
        r"dota/IoU_50_Detail:\s*(?P<detail>\{.*?\})\s+data_time:",
        text,
        flags=re.DOTALL))
    if matches:
        match = matches[-1]
        metrics["mAP"] = float(match.group("map"))
        metrics["AP50"] = float(match.group("ap50"))
        try:
            detail = ast.literal_eval(match.group("detail"))
        except Exception:
            detail = {}
        for cls_name in ("small-vehicle", "large-vehicle", "ship",
                         "bridge", "helipad"):
            cls = detail.get(cls_name, {}) if isinstance(detail, dict) else {}
            prefix = cls_name.replace("-", "_")
            metrics[f"{prefix}_AP"] = as_float(cls.get("ap"))
            metrics[f"{prefix}_recall"] = as_float(cls.get("recall"))
            metrics[f"{prefix}_dets"] = int(as_float(cls.get("num_dets"), 0) or 0)
        return metrics
    return metrics


def parse_losses_from_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [
        line for line in text.splitlines()
        if "Epoch(train)" in line and "[2][" in line
    ]
    out: dict[str, Any] = {}
    if lines:
        line = lines[-1]
        for key in (
                "loss_focus_text_anchor",
                "loss_focus_support_distill",
                "loss_focus_anti",
                "loss_focus_preserve",
                "loss_focus_migration"):
            match = re.search(
                rf"\b{re.escape(key)}:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
                line)
            out[key] = float(match.group(1)) if match else ""
    out.setdefault("loss_focus_eqtext_consistency", "NOT_LOGGED")
    out.setdefault("loss_focus_text_negative_margin", "NOT_LOGGED")
    return out


def parse_geometry(work_dir: Path) -> dict[str, Any]:
    scalars = sorted(work_dir.glob("*/vis_data/scalars.json"))
    out: dict[str, Any] = {
        "dual_text_weight": "",
        "text_delta_norm": "",
        "text_interclass_cos_max": "",
        "text_prompt_collapse_flag": "NOT_LOGGED",
    }
    if not scalars:
        return out
    with scalars[-1].open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for key, value in item.items():
                short = key.split("/")[-1]
                if short in {
                    "dual_text_weight",
                    "text_delta_norm",
                    "text_delta_norm_ratio",
                    "text_interclass_cos_max",
                }:
                    mapped = "text_delta_norm" if short == "text_delta_norm_ratio" else short
                    out[mapped] = value
    cos = as_float(out.get("text_interclass_cos_max"))
    delta = as_float(out.get("text_delta_norm"))
    if cos is not None or delta is not None:
        out["text_prompt_collapse_flag"] = bool(
            (cos is not None and cos >= GEOMETRY_COS_MAX)
            or (delta is not None and delta > GEOMETRY_DELTA_MAX))
    return out


def latest_checkpoint(work_dir: Path) -> str:
    last = work_dir / "last_checkpoint"
    if last.exists():
        value = last.read_text(encoding="utf-8").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = work_dir / path
            if path.exists():
                return str(path)
    ckpts = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else ""


def kill_rules(row: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    sv_dets = int(as_float(row.get("small_vehicle_dets"), 0) or 0)
    mAP = as_float(row.get("mAP"), 0.0) or 0.0
    sv_ap = as_float(row.get("small_vehicle_AP"), 0.0) or 0.0
    if sv_dets > FAIL_SV_DETS_MULT * FOCUS_EPOCH2_SV_DETS:
        failures.append("FAIL_MASSIVE_SV_EXPLOSION")
    if mAP < FAIL_MAP_MIN:
        failures.append("FAIL_AP_COLLAPSE")
    if sv_ap < FAIL_SV_AP_MIN:
        failures.append("FAIL_SV_AP_COLLAPSE")
    cos = as_float(row.get("text_interclass_cos_max"))
    delta = as_float(row.get("text_delta_norm"))
    if ((cos is not None and cos >= GEOMETRY_COS_MAX)
            or (delta is not None and delta > GEOMETRY_DELTA_MAX)):
        failures.append("FAIL_TEXT_GEOMETRY_COLLAPSE")
    return failures


def command_for(config_path: Path, work_dir: Path,
                port: int) -> list[str]:
    return [
        "rtk",
        "env",
        "CUDA_VISIBLE_DEVICES=6,9",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "PYTHONNOUSERSITE=1",
        "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512",
        f"PORT={port}",
        f"PYTHON={OPENRSD_PYTHON}",
        "bash",
        "tools/my_dist_train.sh",
        str(config_path),
        "2",
        "--work-dir",
        str(work_dir),
    ]


def build_reference_row(variant: dict[str, Any]) -> dict[str, Any]:
    ref = variant.get("reference_metrics", {})
    if not ref:
        return {
            "variant_id": variant["variant_id"],
            "status": "SKIPPED_METADATA_ONLY",
            "returncode": 0,
            "mAP": "",
            "AP50": "",
            "small_vehicle_AP": "",
            "small_vehicle_recall": "",
            "small_vehicle_dets": "",
            "large_vehicle_AP": "",
            "ship_AP": "",
            "bridge_AP": "",
            "bridge_dets": "",
            "helipad_AP": "",
            "helipad_dets": "",
            "det_per_img": "",
            "loss_focus_text_anchor": "NOT_RUN_METADATA_ONLY",
            "loss_focus_eqtext_consistency": "NOT_RUN_METADATA_ONLY",
            "loss_focus_text_negative_margin": "NOT_RUN_METADATA_ONLY",
            "dual_text_weight": "",
            "text_delta_norm": "",
            "text_interclass_cos_max": "",
            "text_prompt_collapse_flag": "NOT_RUN_METADATA_ONLY",
            "kill_rule_failures": "",
            "checkpoint": "",
            "log": "",
        }
    return {
        "variant_id": variant["variant_id"],
        "status": "REFERENCE_ONLY",
        "returncode": 0,
        "mAP": ref.get("mAP", 0.6755),
        "AP50": ref.get("AP50", ref.get("mAP", 0.6755)),
        "small_vehicle_AP": ref.get("small_vehicle_AP", 0.5512),
        "small_vehicle_recall": "",
        "small_vehicle_dets": ref.get("small_vehicle_dets", FOCUS_EPOCH2_SV_DETS),
        "large_vehicle_AP": "",
        "ship_AP": "",
        "bridge_AP": "",
        "bridge_dets": "",
        "helipad_AP": "",
        "helipad_dets": "",
        "det_per_img": "",
        "loss_focus_text_anchor": "NOT_RUN_REFERENCE",
        "loss_focus_eqtext_consistency": "NOT_RUN_REFERENCE",
        "loss_focus_text_negative_margin": "NOT_RUN_REFERENCE",
        "dual_text_weight": 0.0,
        "text_delta_norm": 0.0,
        "text_interclass_cos_max": "",
        "text_prompt_collapse_flag": False,
        "kill_rule_failures": "",
        "checkpoint": "",
        "log": "",
    }


def run_variant(repo_root: Path, exp_dir: Path, variant: dict[str, Any],
                execute: bool, port: int, timeout_s: int) -> dict[str, Any]:
    variant_id = variant["variant_id"]
    if not variant.get("train_required", False):
        return build_reference_row(variant)
    config_path = resolve(repo_root, variant["config_py"])
    work_dir = repo_root / "work_dirs" / "eqtext_strict_epoch2_audit_20260610" / variant_id
    run_log = exp_dir / "logs" / f"{variant_id}_epoch2_direct_val.log"
    scan_failures = safety_scan(config_path)
    cmd = command_for(config_path, work_dir, port)
    row: dict[str, Any] = {
        "variant_id": variant_id,
        "config": str(config_path),
        "work_dir": str(work_dir),
        "command": " ".join(cmd),
        "safety_scan_failures": "; ".join(scan_failures),
        "log": str(run_log),
    }
    if scan_failures:
        row.update({
            "status": "BLOCKED_CONFIG_SAFETY_SCAN",
            "returncode": 2,
            "kill_rule_failures": "CONFIG_SAFETY_SCAN",
        })
        return row
    if not execute:
        row.update({
            "status": "DRY_RUN_COMMAND_READY",
            "returncode": 0,
            "kill_rule_failures": "",
        })
        return row
    returncode = run_command(repo_root, cmd, run_log, timeout_s=timeout_s)
    actual_log = latest_log(work_dir, run_log)
    row["returncode"] = returncode
    row["log"] = str(actual_log)
    row["checkpoint"] = latest_checkpoint(work_dir)
    row.update(parse_metrics_from_log(actual_log))
    row.update(parse_losses_from_log(actual_log))
    row.update(parse_geometry(work_dir))
    row["memory"] = "SEE_NVIDIA_SMI_LOG"
    row["oom_status"] = "OOM" if "out of memory" in (
        actual_log.read_text(encoding="utf-8", errors="replace").lower()
        if actual_log.exists() else "") else ""
    failures = kill_rules(row) if row.get("mAP") != "" else []
    row["kill_rule_failures"] = "; ".join(failures)
    if returncode != 0:
        row["status"] = "FAIL_TRAIN_OR_VAL"
    elif failures:
        row["status"] = "FAIL_KILL_RULE"
    else:
        row["status"] = "PASS_EPOCH2_DIRECT_VAL"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--config-index", type=Path,
                        default=EXP_DIR / "configs" / "eqtext_strict_config_index.csv")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Actually launch training. Default is dry-run only.")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--base-port", type=int, default=29733)
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    config_index = resolve(repo_root, args.config_index)
    rows = read_csv(config_index)
    if not rows:
        raise SystemExit(f"Missing config index: {config_index}")
    variants = []
    wanted = set(args.variants or [])
    for row in rows:
        if wanted and row["variant_id"] not in wanted:
            continue
        payload = load_json(resolve(repo_root, row["config_json"]))
        variants.append(payload)

    results: list[dict[str, Any]] = []
    stopped = False
    for idx, variant in enumerate(variants):
        if stopped:
            results.append({
                "variant_id": variant["variant_id"],
                "status": "SKIPPED_AFTER_PREVIOUS_KILL_RULE",
                "returncode": 0,
            })
            continue
        row = run_variant(
            repo_root,
            exp_dir,
            variant,
            execute=args.execute,
            port=args.base_port + idx,
            timeout_s=args.timeout_s)
        results.append(row)
        if (args.execute and not args.continue_on_fail
                and str(row.get("status", "")).startswith("FAIL")):
            stopped = True

    fields = [
        "variant_id",
        "status",
        "returncode",
        "mAP",
        "AP50",
        "small_vehicle_AP",
        "small_vehicle_recall",
        "small_vehicle_dets",
        "large_vehicle_AP",
        "ship_AP",
        "bridge_AP",
        "bridge_dets",
        "helipad_AP",
        "helipad_dets",
        "det_per_img",
        "loss_focus_text_anchor",
        "loss_focus_eqtext_consistency",
        "loss_focus_text_negative_margin",
        "loss_focus_support_distill",
        "loss_focus_anti",
        "loss_focus_preserve",
        "loss_focus_migration",
        "dual_text_weight",
        "text_delta_norm",
        "text_interclass_cos_max",
        "text_prompt_collapse_flag",
        "kill_rule_failures",
        "oom_status",
        "memory",
        "checkpoint",
        "log",
        "config",
        "work_dir",
        "safety_scan_failures",
        "command",
    ]
    out_csv = exp_dir / "tables" / "eqtext_epoch2_direct_val_results.csv"
    out_json = exp_dir / "tables" / "eqtext_epoch2_direct_val_results.json"
    write_csv(out_csv, results, fields)
    write_json(out_json, {
        "status": (
            "DRY_RUN_READY" if not args.execute else
            ("PASS_EQTEXT_EPOCH2_DIRECT_VAL_RUNNER"
             if all(str(r.get("status", "")).startswith(("PASS", "REFERENCE"))
                    for r in results)
             else "FAIL_EQTEXT_EPOCH2_DIRECT_VAL_RUNNER")),
        "execute": bool(args.execute),
        "stop_on_fail": not args.continue_on_fail,
        "results": results,
    })
    print(json.dumps({
        "status": "DRY_RUN_READY" if not args.execute else "RUN_COMPLETE",
        "execute": bool(args.execute),
        "rows": len(results),
        "results_csv": str(out_csv),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the stopped EQ_V33 epoch2 validation failure."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EXP_DIR = Path("resultmd/exp_eqtext_strict_epoch2_audit_20260610")
DEFAULT_V33_WORK_DIR = Path(
    "work_dirs/focus_eqtext_dota_short_20260609/"
    "EQ_V33_text_eq_clean_dota2_targets")
DEFAULT_COMPARE_DIR = Path("resultmd/exp_focus_ovd_eqv30_val14_compare")
DEFAULT_FOCUS_EXP_DIR = Path("resultmd/exp_focus_ovd_20260608")
FOCUS_EPOCH2_SV_DETS = 347_552
FOCUS_EPOCH2_MAP = 0.6755
FOCUS_EPOCH2_SV_AP = 0.5512
FAIL_MAP_MIN = 0.62
FAIL_SV_AP_MIN = 0.50
FAIL_SV_DETS_MULT = 1.5

SUBDIRS = (
    "preflight",
    "configs",
    "train",
    "eval",
    "tables",
    "figures",
    "reports",
    "logs",
)


def ensure_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


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


def md_table(rows: Sequence[Mapping[str, Any]],
             fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def newest_log(work_dir: Path) -> Path | None:
    logs = sorted(
        list(work_dir.glob("*.log")) + list(work_dir.glob("*/*.log")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    return logs[-1] if logs else None


def config_path_for(work_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    candidates = sorted(work_dir.glob("*.py"))
    return candidates[0] if candidates else None


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_focus_reference(compare_dir: Path) -> dict[str, Any]:
    table = compare_dir / "tables" / "val14_map_key_ap.csv"
    focus = {
        "focus_epoch2_mAP": FOCUS_EPOCH2_MAP,
        "focus_epoch2_svAP": FOCUS_EPOCH2_SV_AP,
        "focus_epoch2_sv_dets": FOCUS_EPOCH2_SV_DETS,
    }
    for row in read_csv(table):
        if str(row.get("epoch")) != "2":
            continue
        focus["focus_epoch2_mAP"] = parse_float(
            row.get("FOCUS-OVD_mAP"), FOCUS_EPOCH2_MAP)
        focus["focus_epoch2_svAP"] = parse_float(
            row.get("FOCUS-OVD_svAP"), FOCUS_EPOCH2_SV_AP)
        focus["focus_epoch2_sv_dets"] = int(parse_float(
            row.get("FOCUS-OVD_sv_dets"), FOCUS_EPOCH2_SV_DETS) or 0)
        focus["focus_epoch2_lvAP"] = parse_float(row.get("FOCUS-OVD_lvAP"))
        focus["focus_epoch2_shipAP"] = parse_float(row.get("FOCUS-OVD_shipAP"))
        break
    return focus


def parse_val_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    val_matches = list(re.finditer(
        r"Epoch\(val\) \[2\]\[\s*\d+/\d+\].*?dota/mAP:\s*"
        r"(?P<map>[-+]?\d+(?:\.\d+)?)\s+dota/AP50:\s*"
        r"(?P<ap50>[-+]?\d+(?:\.\d+)?).*?"
        r"dota/IoU_50_Detail:\s*(?P<detail>\{.*?\})\s+data_time:",
        log_text,
        flags=re.DOTALL,
    ))
    if val_matches:
        match = val_matches[-1]
        metrics["epoch2_mAP"] = float(match.group("map"))
        metrics["epoch2_AP50"] = float(match.group("ap50"))
        try:
            detail = ast.literal_eval(match.group("detail"))
        except Exception:
            detail = {}
        for cls_name in ("small-vehicle", "bridge", "helipad",
                         "large-vehicle", "ship"):
            cls = detail.get(cls_name, {}) if isinstance(detail, dict) else {}
            prefix = cls_name.replace("-", "_")
            metrics[f"{prefix}_AP"] = parse_float(cls.get("ap"))
            metrics[f"{prefix}_recall"] = parse_float(cls.get("recall"))
            metrics[f"{prefix}_dets"] = int(parse_float(
                cls.get("num_dets"), 0) or 0)
            metrics[f"{prefix}_gts"] = int(parse_float(
                cls.get("num_gts"), 0) or 0)
        return metrics

    # Fallback for logs that contain only the pretty AP table.
    table_rows = re.findall(
        r"\|\s*([a-zA-Z0-9-]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        log_text)
    for cls_name, gts, dets, recall, ap in table_rows:
        prefix = cls_name.replace("-", "_")
        metrics[f"{prefix}_AP"] = float(ap)
        metrics[f"{prefix}_recall"] = float(recall)
        metrics[f"{prefix}_dets"] = int(dets)
        metrics[f"{prefix}_gts"] = int(gts)
    map_rows = re.findall(r"\|\s*mAP\s*\|\s*\|\s*\|\s*\|\s*([0-9.]+)\s*\|",
                          log_text)
    if map_rows:
        metrics["epoch2_mAP"] = float(map_rows[-1])
        metrics["epoch2_AP50"] = metrics["epoch2_mAP"]
    return metrics


def parse_epoch2_losses(log_text: str) -> dict[str, Any]:
    lines = [
        line for line in log_text.splitlines()
        if "Epoch(train)" in line and "[2][" in line
    ]
    if not lines:
        return {}
    line = lines[-1]
    keys = [
        "loss_cls",
        "loss_bbox",
        "loss_focus_support_distill",
        "loss_focus_text_anchor",
        "loss_focus_anti",
        "loss_focus_preserve",
        "loss_focus_migration",
        "loss_focus_total",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        match = re.search(rf"\b{re.escape(key)}:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
                          line)
        out[key] = float(match.group(1)) if match else ""
    out["epoch2_loss_source"] = "last_epoch2_train_log_line"
    return out


def parse_scalars(work_dir: Path) -> dict[str, Any]:
    scalar_files = list(work_dir.glob("*/vis_data/scalars.json"))
    if not scalar_files:
        return {}
    path = sorted(scalar_files, key=lambda p: p.stat().st_mtime)[-1]
    wanted = {
        "text_delta_norm",
        "text_delta_norm_ratio",
        "text_interclass_cos",
        "text_interclass_cos_max",
        "dual_text_weight",
    }
    latest: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for key, value in item.items():
                if key in wanted or any(key.endswith("/" + w) for w in wanted):
                    latest[key.split("/")[-1]] = value
    latest["scalars_path"] = str(path)
    return latest


def parse_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {
            "support_type": "",
            "with_aux_bbox_head": "",
            "dual_text_weight": "",
            "config_path": "",
        }
    text = config_path.read_text(encoding="utf-8")
    support_matches = re.findall(r"support_type\s*=\s*['\"]([^'\"]+)['\"]", text)
    support_type = support_matches[-1] if support_matches else ""
    if re.search(r"with_aux_bbox_head\s*=\s*False", text):
        with_aux = False
    elif re.search(r"with_aux_bbox_head\s*=\s*True", text):
        with_aux = True
    else:
        # V33 inherits the full A10/FOCUS auxiliary head.
        with_aux = True
    weight_match = re.search(
        r"text_weight_init\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    max_weight_match = re.search(
        r"max_text_weight\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    return {
        "support_type": support_type,
        "with_aux_bbox_head": with_aux,
        "dual_text_weight": (
            float(weight_match.group(1)) if weight_match else ""),
        "dual_text_weight_max": (
            float(max_weight_match.group(1)) if max_weight_match else ""),
        "use_declip_support": bool(re.search(
            r"use_declip_support\s*=\s*True", text)),
        "config_path": str(config_path),
    }


def classify(row: dict[str, Any]) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    mAP = parse_float(row.get("epoch2_mAP"), 0.0) or 0.0
    sv_ap = parse_float(row.get("small_vehicle_AP"), 0.0) or 0.0
    sv_dets = int(parse_float(row.get("small_vehicle_dets"), 0.0) or 0)
    focus_sv_dets = int(row.get("focus_epoch2_sv_dets") or FOCUS_EPOCH2_SV_DETS)
    if sv_ap < FAIL_SV_AP_MIN:
        reasons.append("small_vehicle_AP < 0.50")
    if mAP < FAIL_MAP_MIN:
        reasons.append("mAP < 0.62")
    if sv_dets > FAIL_SV_DETS_MULT * focus_sv_dets:
        reasons.append("small_vehicle_dets > 1.5x FOCUS epoch2")
    if reasons:
        return "FAILED_TEXT_CALIBRATION_EPOCH2", True, reasons
    return "PASS_EPOCH2_TEXT_CALIBRATION_GATE", False, []


def write_report(path: Path, row: dict[str, Any],
                 fail_reasons: Sequence[str]) -> None:
    fields = [
        "run_id",
        "epoch2_mAP",
        "epoch2_AP50",
        "small_vehicle_AP",
        "small_vehicle_recall",
        "small_vehicle_dets",
        "bridge_AP",
        "bridge_dets",
        "helipad_AP",
        "helipad_dets",
        "loss_cls",
        "loss_bbox",
        "loss_focus_support_distill",
        "loss_focus_text_anchor",
        "loss_focus_anti",
        "loss_focus_preserve",
        "loss_focus_migration",
        "support_type",
        "with_aux_bbox_head",
        "dual_text_weight",
        "text_delta_norm",
        "text_interclass_cos_max",
        "status",
        "recommend_stop",
    ]
    lines = [
        "# EQ_V33 Epoch2 Failure Audit",
        "",
        "## Verdict",
        "",
        f"- status: `{row['status']}`",
        f"- recommend_stop: `{row['recommend_stop']}`",
        f"- failure reasons: `{'; '.join(fail_reasons) if fail_reasons else 'none'}`",
        "",
        "V33 is not a final denial of text-side equivariance. It is a current direct text/logit fusion calibration failure at epoch2.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(md_table([row], fields))
    lines.extend([
        "",
        "## Reference Thresholds",
        "",
        f"- FOCUS epoch2 mAP: `{row.get('focus_epoch2_mAP')}`",
        f"- FOCUS epoch2 small-vehicle AP: `{row.get('focus_epoch2_svAP')}`",
        f"- FOCUS epoch2 small-vehicle dets: `{row.get('focus_epoch2_sv_dets')}`",
        f"- kill rule mAP minimum: `{FAIL_MAP_MIN}`",
        f"- kill rule small-vehicle AP minimum: `{FAIL_SV_AP_MIN}`",
        f"- kill rule small-vehicle det limit: `{int(FAIL_SV_DETS_MULT * int(row.get('focus_epoch2_sv_dets') or FOCUS_EPOCH2_SV_DETS))}`",
        "",
        "## Interpretation",
        "",
        "- EQ_V30 was a dirty failure because it used visual support, disabled the aux branch, and used mismatched DOTA1 targets.",
        "- V33 repaired those structural issues, but epoch2 still has massive small-vehicle over-detection and low AP.",
        "- Therefore the next step must be layered TEXT-side validation: zero/shadow, loss-only, then tiny fusion.",
        "- Do not continue V33 as a long run.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--v33-work-dir", type=Path, default=DEFAULT_V33_WORK_DIR)
    parser.add_argument("--v33-config", type=Path, default=None)
    parser.add_argument("--v33-log", type=Path, default=None)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--focus-exp-dir", type=Path, default=DEFAULT_FOCUS_EXP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.output_dir)
    ensure_tree(exp_dir)
    work_dir = resolve(repo_root, args.v33_work_dir)
    log_path = (resolve(repo_root, args.v33_log)
                if args.v33_log else newest_log(work_dir))
    if log_path is None or not log_path.exists():
        raise SystemExit(f"Cannot find V33 log under {work_dir}")
    config_path = config_path_for(
        work_dir,
        resolve(repo_root, args.v33_config) if args.v33_config else None)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    row: dict[str, Any] = {
        "run_id": "EQ_V33_text_eq_clean_dota2_targets",
        "v33_work_dir": str(work_dir),
        "v33_log": str(log_path),
        "focus_exp_dir": str(resolve(repo_root, args.focus_exp_dir)),
    }
    row.update(parse_focus_reference(resolve(repo_root, args.compare_dir)))
    row.update(parse_val_metrics(log_text))
    row.update(parse_epoch2_losses(log_text))
    row.update(parse_config(config_path))
    scalars = parse_scalars(work_dir)
    row["text_delta_norm"] = scalars.get(
        "text_delta_norm_ratio", scalars.get("text_delta_norm", ""))
    row["text_interclass_cos"] = scalars.get("text_interclass_cos", "")
    row["text_interclass_cos_max"] = scalars.get("text_interclass_cos_max", "")
    if "dual_text_weight" in scalars:
        row["dual_text_weight"] = scalars["dual_text_weight"]
    status, recommend_stop, reasons = classify(row)
    row["status"] = status
    row["recommend_stop"] = recommend_stop
    row["failure_reasons"] = "; ".join(reasons)
    row["note"] = (
        "V33 is a direct text/logit fusion calibration failure, not a final "
        "denial of text-side equivariance.")

    fields = [
        "run_id",
        "epoch2_mAP",
        "epoch2_AP50",
        "small_vehicle_AP",
        "small_vehicle_recall",
        "small_vehicle_dets",
        "large_vehicle_AP",
        "ship_AP",
        "bridge_AP",
        "bridge_dets",
        "helipad_AP",
        "helipad_dets",
        "loss_cls",
        "loss_bbox",
        "loss_focus_support_distill",
        "loss_focus_text_anchor",
        "loss_focus_anti",
        "loss_focus_preserve",
        "loss_focus_migration",
        "support_type",
        "with_aux_bbox_head",
        "dual_text_weight",
        "dual_text_weight_max",
        "text_delta_norm",
        "text_interclass_cos",
        "text_interclass_cos_max",
        "focus_epoch2_mAP",
        "focus_epoch2_svAP",
        "focus_epoch2_sv_dets",
        "status",
        "recommend_stop",
        "failure_reasons",
        "note",
        "v33_log",
        "config_path",
    ]
    write_csv(exp_dir / "tables" / "v33_epoch2_failure_audit.csv",
              [row], fields)
    write_json(exp_dir / "tables" / "v33_epoch2_failure_audit.json", row)
    write_report(exp_dir / "reports" / "v33_epoch2_failure_audit.md",
                 row, reasons)
    print(json.dumps({
        "status": row["status"],
        "recommend_stop": row["recommend_stop"],
        "epoch2_mAP": row.get("epoch2_mAP"),
        "small_vehicle_AP": row.get("small_vehicle_AP"),
        "small_vehicle_dets": row.get("small_vehicle_dets"),
        "report": str(exp_dir / "reports" / "v33_epoch2_failure_audit.md"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

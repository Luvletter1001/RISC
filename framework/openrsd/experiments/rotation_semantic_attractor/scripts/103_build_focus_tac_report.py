#!/usr/bin/env python3
"""Build the FOCUS-TAC markdown/html report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import torch


KEY_CLASSES = (
    "small_vehicle",
    "large_vehicle",
    "ship",
    "bridge",
    "helipad",
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("variant_id")): row for row in rows}


def resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def checkpoint_for_variant(exp_dir: Path, variant_id: str) -> Path | None:
    candidates: list[Path] = []
    for summary_path in sorted(exp_dir.glob("train*/focus_tac_train_summary.json")):
        train_summary = read_json(summary_path, {})
        for row in train_summary.get("variants", []):
            if row.get("variant_id") == variant_id and row.get("checkpoint"):
                path = Path(row["checkpoint"])
                if path.exists():
                    candidates.append(path)
    candidates.extend(exp_dir.glob(f"train*/{variant_id}/work_dir/epoch_*.pth"))
    if candidates:
        return sorted(set(candidates), key=checkpoint_sort_key, reverse=True)[0]
    return None


def checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"epoch_(\d+)\.pth$", path.name)
    epoch = int(match.group(1)) if match else -1
    return epoch, path.stat().st_mtime


def collect_eval_rows(exp_dir: Path) -> list[dict[str, Any]]:
    """Collect PASS eval rows from all FOCUS-TAC eval summary directories."""
    rows_by_variant: dict[str, tuple[float, dict[str, Any]]] = {}
    for summary_path in sorted(exp_dir.glob("eval*/focus_tac_eval_summary.json")):
        payload = read_json(summary_path, {})
        mtime = summary_path.stat().st_mtime
        for row in payload.get("rows", []):
            variant_id = str(row.get("variant_id", ""))
            if not variant_id:
                continue
            # Prefer the fixed-seed zero check for V00/V01 over older ad-hoc
            # summaries, prefer PASS metrics over partial rows, then otherwise
            # keep the newest row so incomplete variants remain visible.
            priority = mtime
            if row.get("eval_status") == "PASS":
                priority += 1_000_000
            if "eval_seed20260610" in str(summary_path):
                priority += 10_000_000
            current = rows_by_variant.get(variant_id)
            if current is None or priority >= current[0]:
                rows_by_variant[variant_id] = (priority, row)
    desired_order = [
        "TAC_V00_FOCUS_ep24_eval",
        "TAC_V01_FOCUS_TAC_zero",
        "TAC_V03_FOCUS_TAC_alpha_only_monitored",
        "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
    ]
    ordered: list[dict[str, Any]] = []
    for variant_id in desired_order:
        if variant_id in rows_by_variant:
            ordered.append(rows_by_variant[variant_id][1])
    for variant_id in sorted(set(rows_by_variant) - set(desired_order)):
        ordered.append(rows_by_variant[variant_id][1])
    return ordered


def extract_alpha_beta(
        repo_root: Path,
        exp_dir: Path,
        variant_id: str,
        config_json: Path | None) -> dict[str, Any]:
    payload = read_json(config_json, {}) if config_json else {}
    tac_cfg = payload.get("tac", {})
    alpha_bound = float(tac_cfg.get("alpha_bound", 0.05))
    beta_bound = float(tac_cfg.get("beta_bound", 0.20))
    checkpoint = checkpoint_for_variant(exp_dir, variant_id)
    raw_alpha = None
    raw_beta = None
    if checkpoint is not None:
        ckpt = torch.load(str(checkpoint), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        for key, value in state.items():
            clean_key = key.replace("module.", "")
            if clean_key.endswith("bbox_head.focus_text_anchor_calibration.raw_alpha"):
                raw_alpha = value.float()
            if clean_key.endswith("bbox_head.focus_text_anchor_calibration.raw_beta"):
                raw_beta = value.float()
    if raw_alpha is None:
        raw_alpha = torch.zeros(18)
    if raw_beta is None:
        raw_beta = torch.zeros_like(raw_alpha)
    use_beta = bool(tac_cfg.get("use_beta", False))
    alpha = alpha_bound * torch.tanh(raw_alpha)
    beta = beta_bound * torch.tanh(raw_beta) if use_beta else torch.zeros_like(raw_beta)
    return {
        "variant_id": variant_id,
        "checkpoint": str(checkpoint or ""),
        "max_abs_alpha": float(alpha.abs().max().item()) if alpha.numel() else 0.0,
        "max_abs_beta": float(beta.abs().max().item()) if beta.numel() else 0.0,
        "alpha_values": [float(v) for v in alpha.tolist()],
        "beta_values": [float(v) for v in beta.tolist()],
    }


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = by_variant(rows)
    base = lookup.get("TAC_V00_FOCUS_ep24_eval", {})
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("eval_status") != "PASS":
            delta = {
                "variant_id": row.get("variant_id", ""),
                "mAP": row.get("mAP", ""),
                "mAP_delta": "",
                "AP50": row.get("AP50", ""),
            }
            for cls in KEY_CLASSES:
                delta[f"{cls}_AP"] = row.get(f"{cls}_AP", "")
                delta[f"{cls}_AP_delta"] = ""
                delta[f"{cls}_dets"] = row.get(f"{cls}_dets", "")
                delta[f"{cls}_dets_delta"] = ""
            out.append(delta)
            continue
        delta = {
            "variant_id": row.get("variant_id", ""),
            "mAP": row.get("mAP", ""),
            "mAP_delta": as_float(row, "mAP") - as_float(base, "mAP"),
            "AP50": row.get("AP50", ""),
        }
        for cls in KEY_CLASSES:
            delta[f"{cls}_AP"] = row.get(f"{cls}_AP", "")
            delta[f"{cls}_AP_delta"] = (
                as_float(row, f"{cls}_AP")
                - as_float(base, f"{cls}_AP"))
            delta[f"{cls}_dets"] = row.get(f"{cls}_dets", "")
            delta[f"{cls}_dets_delta"] = (
                as_float(row, f"{cls}_dets")
                - as_float(base, f"{cls}_dets"))
        out.append(delta)
    return out


def verdict_for_variant(row: dict[str, Any],
                        base: dict[str, Any],
                        alpha_beta: dict[str, Any]) -> dict[str, Any]:
    if row.get("eval_status") != "PASS":
        return {
            "variant_id": row.get("variant_id"),
            "recommendation": "NEED_LONGER_RUN",
            "reason": "missing full eval metrics",
        }
    mAP_delta = as_float(row, "mAP") - as_float(base, "mAP")
    sv_delta = as_float(row, "small_vehicle_AP") - as_float(
        base, "small_vehicle_AP")
    sv_dets = as_float(row, "small_vehicle_dets")
    base_sv_dets = as_float(base, "small_vehicle_dets")
    bridge_dets = as_float(row, "bridge_dets")
    base_bridge_dets = as_float(base, "bridge_dets")
    helipad_dets = as_float(row, "helipad_dets")
    base_helipad_dets = as_float(base, "helipad_dets")
    det_explosion = (
        (base_sv_dets > 0 and sv_dets > base_sv_dets * 1.05)
        or (base_bridge_dets > 0 and bridge_dets > base_bridge_dets * 1.05)
        or (base_helipad_dets > 0 and helipad_dets > base_helipad_dets * 1.05))
    saturated = (
        float(alpha_beta.get("max_abs_alpha", 0.0)) >= 0.049
        or float(alpha_beta.get("max_abs_beta", 0.0)) >= 0.195)
    keep = (
        (mAP_delta >= 0.002 or (abs(mAP_delta) < 1e-12 and sv_delta >= 0.005))
        and sv_delta >= 0.0
        and not det_explosion
        and not saturated)
    if keep:
        recommendation = "KEEP"
    elif mAP_delta < 0.0 or sv_delta < 0.0 or det_explosion or saturated:
        recommendation = "DISCARD"
    else:
        recommendation = "NEED_LONGER_RUN"
    return {
        "variant_id": row.get("variant_id"),
        "recommendation": recommendation,
        "mAP_delta": mAP_delta,
        "small_vehicle_AP_delta": sv_delta,
        "det_explosion": det_explosion,
        "alpha_beta_saturated": saturated,
    }


def build_report(repo_root: Path, exp_dir: Path) -> dict[str, Any]:
    rows = collect_eval_rows(exp_dir)
    lookup = by_variant(rows)
    config_rows = {}
    config_index = exp_dir / "configs/focus_tac_variant_config_index.csv"
    if config_index.exists():
        with config_index.open("r", encoding="utf-8", newline="") as f:
            config_rows = {r["variant_id"]: r for r in csv.DictReader(f)}

    alpha_rows = []
    for variant_id in lookup:
        config_json = None
        if variant_id in config_rows:
            config_json = resolve(repo_root, config_rows[variant_id]["config_json"])
        alpha_rows.append(extract_alpha_beta(
            repo_root, exp_dir, variant_id, config_json))
    alpha_lookup = by_variant(alpha_rows)
    base = lookup.get("TAC_V00_FOCUS_ep24_eval", {})
    zero = lookup.get("TAC_V01_FOCUS_TAC_zero", {})
    zero_equivalent = (
        base.get("eval_status") == "PASS"
        and zero.get("eval_status") == "PASS"
        and as_float(base, "mAP") == as_float(zero, "mAP")
        and as_float(base, "small_vehicle_AP") == as_float(
            zero, "small_vehicle_AP")
        and as_float(base, "small_vehicle_dets") == as_float(
            zero, "small_vehicle_dets"))
    verdicts = [
        verdict_for_variant(row, base, alpha_lookup.get(row["variant_id"], {}))
        for row in rows
        if row.get("variant_id") not in {
            "TAC_V00_FOCUS_ep24_eval",
            "TAC_V01_FOCUS_TAC_zero",
        }
    ]
    final_recommendation = "NEED_LONGER_RUN"
    if any(v.get("recommendation") == "KEEP" for v in verdicts):
        final_recommendation = "KEEP"
    elif verdicts and all(v.get("recommendation") == "DISCARD" for v in verdicts):
        final_recommendation = "DISCARD"
    return {
        "status": "PASS_FOCUS_TAC_REPORT",
        "zero_equivalent": zero_equivalent,
        "final_recommendation": final_recommendation,
        "eval_rows": rows,
        "delta_rows": build_delta_rows(rows),
        "alpha_beta_rows": alpha_rows,
        "variant_verdicts": verdicts,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metric_fields = [
        "variant_id",
        "eval_status",
        "mAP",
        "AP50",
        "small_vehicle_AP",
        "large_vehicle_AP",
        "ship_AP",
        "bridge_AP",
        "helipad_AP",
        "small_vehicle_dets",
        "bridge_dets",
        "helipad_dets",
    ]
    delta_fields = [
        "variant_id",
        "mAP_delta",
        "small_vehicle_AP_delta",
        "large_vehicle_AP_delta",
        "ship_AP_delta",
        "bridge_AP_delta",
        "helipad_AP_delta",
        "small_vehicle_dets_delta",
        "bridge_dets_delta",
        "helipad_dets_delta",
    ]
    alpha_fields = [
        "variant_id",
        "max_abs_alpha",
        "max_abs_beta",
        "checkpoint",
    ]
    verdict_fields = [
        "variant_id",
        "recommendation",
        "mAP_delta",
        "small_vehicle_AP_delta",
        "det_explosion",
        "alpha_beta_saturated",
        "reason",
    ]
    lines = [
        "# FOCUS-TAC Report",
        "",
        "## Why minimal TAC",
        "",
        "FOCUS-TAC only calibrates class text-support logits with a bounded class-wise scale and bias. It keeps the FOCUS-OVD support path, support bank, support residual adapter, detector head, backbone, neck, and box regression path unchanged.",
        "",
        "## Why no support replacement",
        "",
        "Direct support replacement previously caused severe support-space mismatch. This report therefore treats support replacement as forbidden and leaves native OpenRSD text support untouched.",
        "",
        "## Why no dual fusion",
        "",
        "Dual text/visual fusion changed score calibration in prior EQText runs. FOCUS-TAC calibrates logits after contrastive classification instead of mixing support embeddings.",
        "",
        "## Zero equivalence result",
        "",
        f"- zero equivalent: `{report['zero_equivalent']}`",
        "",
        "## AP/mAP comparison",
        "",
    ]
    lines.extend(md_table(report["eval_rows"], metric_fields))
    lines.extend(["", "## Per-class AP and det deltas", ""])
    lines.extend(md_table(report["delta_rows"], delta_fields))
    lines.extend(["", "## Alpha/Beta learned values", ""])
    lines.extend(md_table(report["alpha_beta_rows"], alpha_fields))
    lines.extend(["", "## Variant verdicts", ""])
    lines.extend(md_table(report["variant_verdicts"], verdict_fields))
    lines.extend([
        "",
        "## Final recommendation",
        "",
        f"- recommendation: `{report['final_recommendation']}`",
        "",
        "## Safety notes",
        "",
        "- DeCLIP support was not enabled.",
        "- Support bank was not replaced.",
        "- support_type remains text.",
        "- with_aux_bbox_head is not disabled by TAC configs.",
        "- Text encoder is not trained.",
        "- DOTA1 val targets and unverified DOTA2 hard negatives are not used.",
        "- Non-TAC parameters are frozen for TAC train variants by trainable allowlist.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(md_path: Path, html_path: Path) -> None:
    body = html.escape(md_path.read_text(encoding="utf-8"))
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>FOCUS-TAC Report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;}"
        "pre{white-space:pre-wrap;line-height:1.4}</style><pre>"
        + body + "</pre>",
        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    exp_dir = args.exp_dir
    if not exp_dir.is_absolute():
        exp_dir = repo_root / exp_dir
    report = build_report(repo_root, exp_dir)
    report_md = exp_dir / "focus_tac_report.md"
    report_html = exp_dir / "focus_tac_report.html"
    write_json(exp_dir / "focus_tac_report.json", report)
    write_csv(exp_dir / "tables/focus_tac_eval_rows.csv", report["eval_rows"])
    write_csv(exp_dir / "tables/focus_tac_delta_rows.csv", report["delta_rows"])
    write_csv(exp_dir / "tables/focus_tac_alpha_beta_rows.csv", report["alpha_beta_rows"])
    write_markdown(report_md, report)
    write_html(report_md, report_html)
    print(json.dumps({
        "status": report["status"],
        "zero_equivalent": report["zero_equivalent"],
        "recommendation": report["final_recommendation"],
        "report_md": str(report_md),
        "report_html": str(report_html),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

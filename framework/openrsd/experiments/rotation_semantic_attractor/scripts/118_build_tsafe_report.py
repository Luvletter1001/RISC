#!/usr/bin/env python3
"""Build final FOCUS-T-Safe report."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import EXP_DIR, FOCUS_REF, read_csv, read_json, resolve, write_json


def _status(payload: dict[str, Any], default: str = "NOT_RUN") -> str:
    return str(payload.get("status") or default)


def _first_table_value(path: Path, key: str) -> str:
    rows = read_csv(path)
    if not rows:
        return ""
    return str(rows[0].get(key, ""))


def build_markdown(exp_dir: Path, payload: dict[str, Any]) -> str:
    lines = [
        "# FOCUS-T-Safe Final Report",
        "",
        "## Answers",
        "",
        f"1. Did zero disturbance pass? `{payload['zero_disturbance_status']}`",
        f"2. Does shadow text have discriminative signal? `{payload['shadow_status']}` with combined AUC `{payload['shadow_auc']}`",
        f"3. Is text side safe as a gate? `{payload['text_safe_as_gate']}`",
        f"4. Did loss-only remain non-invasive? `{payload['loss_only_status']}`",
        f"5. Did offline tiny calibration help? `{payload['offline_calibration_status']}`",
        f"6. Should text side enter the main method? `{payload['text_side_main_method_decision']}`",
        f"7. Does FOCUS-OVD remain main method? `YES`",
        "",
        "## Guardrails",
        "",
        "- Text side does not directly affect final logits.",
        "- Native support bank is not replaced.",
        "- DeCLIP support is not used.",
        "- Anti-attractor training is not enabled.",
        "- No AP improvement from text side is claimed.",
        "",
        "## FOCUS Reference",
        "",
        f"- FOCUS-OVD ep24 mAP: `{FOCUS_REF['mAP']}`",
        f"- FOCUS-OVD ep24 small-vehicle AP: `{FOCUS_REF['small_vehicle_AP']}`",
        "",
    ]
    report_path = exp_dir / "reports" / "focus_tsafe_final_report.md"
    lines.extend([
        "## Artifact Paths",
        "",
        f"- final report: `{report_path}`",
        f"- zero disturbance: `{exp_dir / 'preflight' / 'tsafe_zero_disturbance_report.md'}`",
        f"- shadow scores: `{exp_dir / 'shadow_scores' / 'tsafe_shadow_scores.csv'}`",
        f"- shadow analysis: `{exp_dir / 'reports' / 'tsafe_shadow_signal_report.md'}`",
        f"- loss-only smoke: `{exp_dir / 'loss_only' / 'tsafe_loss_only_report.md'}`",
        f"- offline calibration: `{exp_dir / 'offline_mixing' / 'tsafe_tiny_calibration_report.md'}`",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    zero = read_json(exp_dir / "preflight" / "tsafe_zero_disturbance_report.json", {})
    shadow = read_json(exp_dir / "reports" / "tsafe_shadow_signal_report.json", {})
    loss = read_json(exp_dir / "loss_only" / "tsafe_loss_only_report.json", {})
    offline = read_json(exp_dir / "offline_mixing" / "tsafe_tiny_calibration_report.json", {})
    shadow_status = _status(shadow)
    payload = {
        "zero_disturbance_status": _status(zero),
        "shadow_status": shadow_status,
        "shadow_auc": shadow.get("combined_auc", ""),
        "degenerate_high_rate": shadow.get("degenerate_high_rate", ""),
        "padding_high_rate": shadow.get("padding_high_rate", ""),
        "category_means_path": str(exp_dir / "tables" / "tsafe_shadow_category_means.csv"),
        "loss_only_status": _status(loss),
        "offline_calibration_status": _status(offline),
        "selected_lambda": offline.get("selected_lambda", ""),
        "text_side_can_affect_logits": False,
        "focus_ovd_remains_main_method": True,
        "text_safe_as_gate": (
            "YES" if shadow_status == "SHADOW_SIGNAL_PASS"
            and safe_shadow(shadow) else "NO"),
        "text_side_main_method_decision": "NO_SHADOW_OR_OFFLINE_ONLY",
        "focus_reference": FOCUS_REF,
    }
    md = build_markdown(exp_dir, payload)
    md_path = exp_dir / "reports" / "focus_tsafe_final_report.md"
    html_path = exp_dir / "reports" / "focus_tsafe_final_report.html"
    html_alt = exp_dir / "html" / "focus_tsafe_final_report.html"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    html_body = "<pre>" + html.escape(md) + "</pre>\n"
    html_path.write_text(html_body, encoding="utf-8")
    html_alt.parent.mkdir(parents=True, exist_ok=True)
    html_alt.write_text(html_body, encoding="utf-8")
    write_json(exp_dir / "reports" / "focus_tsafe_final_report.json", payload)
    print(json.dumps({**payload, "report": str(md_path)}, indent=2, ensure_ascii=False))
    return 0


def safe_shadow(shadow: dict[str, Any]) -> bool:
    try:
        return (float(shadow.get("degenerate_high_rate", 1.0)) <= 0.30
                and float(shadow.get("padding_high_rate", 1.0)) <= 0.30
                and not bool(shadow.get("text_interclass_cos_collapse_flag", True)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())

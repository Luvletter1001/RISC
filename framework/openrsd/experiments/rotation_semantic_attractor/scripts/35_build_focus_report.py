#!/usr/bin/env python3
"""Build a compact FOCUS-OVD method and artifact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("resultmd/exp_focus_ovd_20260608")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    audit = read_json(args.root / "audit_labels/focus_audit_label_summary.json")
    equivalence = read_json(args.root / "baseline_equivalence/equivalence_smoke.json")
    orientation = read_json(args.root / "orientation_probe/orientation_probe_summary.json")
    train = read_json(args.root / "train_smoke/train_smoke_plan.json")
    eval_plan = read_json(args.root / "eval_smoke/eval_smoke_plan.json")
    ablation = read_json(args.root / "ablation_plan/focus_ablation_plan.json")

    md = [
        "# FOCUS-OVD Report",
        "",
        "## Method",
        "",
        "FOCUS-OVD adds a Fourier orientation-conditioned, zero-initialized residual adapter on top of the native OpenRSD support bank. The first-stage target is `small-vehicle`; DeCLIP support, direct prompt replacement, box regression, NMS, and postprocess are intentionally untouched.",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["Dense feature map"] --> B["Fourier orientation learner"]',
        '  B --> C["Periodic orientation code"]',
        '  D["Native OpenRSD support bank"] --> E["Visual support mapping"]',
        '  C --> F["Zero-init residual support adapter"]',
        '  E --> F',
        '  F --> G["Conditioned support logits"]',
        '  G --> H["Original OpenRSD class-score aggregation"]',
        "```",
        "",
        "## Artifact Status",
        "",
        f"- Audit labels: `{audit.get('output_csv', audit.get('status'))}`",
        f"- Baseline equivalence: {equivalence.get('status')} (max diff={equivalence.get('max_abs_diff')})",
        f"- Orientation probe: {orientation.get('status')} (mean error={orientation.get('mean_periodic_error_degrees')})",
        f"- Train smoke: {train.get('status')}",
        f"- Eval smoke: {eval_plan.get('status')}",
        f"- Ablations: {len(ablation.get('ablations', []))}",
        "",
        "## Corrected-FSV Policy",
        "",
        "- `corrected_false_sv` rows are hard negatives.",
        "- `annotation_missing_true_vehicle` rows are preserve positives, not negatives.",
        "- Ambiguous/invalid rows are excluded.",
        "- Degenerate large-SV and padding rows are excluded from hard negatives.",
        "",
        "## Submission-Level Claim",
        "",
        "The publishable idea is not direction-as-text. It is support-space conditional calibration: local Fourier evidence modulates only the relevant native support vectors through a norm-bounded residual, preserving open-vocabulary semantics while reducing orientation-driven false small-vehicle attraction.",
    ]
    args.root.mkdir(parents=True, exist_ok=True)
    report_md = args.root / "focus_ovd_report.md"
    report_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    html = "<html><body>" + "\n".join(
        f"<p>{line}</p>" if line and not line.startswith("#") else f"<h1>{line.lstrip('# ').strip()}</h1>"
        for line in md if not line.startswith("```") and not line.startswith("flowchart")
    ) + "</body></html>\n"
    (args.root / "focus_ovd_report.html").write_text(html, encoding="utf-8")
    print(str(report_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

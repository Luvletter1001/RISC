#!/usr/bin/env python3
"""Build the FOCUS-OVD ablation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/ablation_plan")


ABLATIONS = [
    ("A0_baseline", "native OpenRSD, no FOCUS modules", "establish exact baseline"),
    ("A1_focus_sv_zero_init", "FOCUS enabled, alpha=0", "baseline-equivalence control"),
    ("A2_focus_sv_trainable", "small-vehicle residual adapter", "main method"),
    ("A3_no_confidence_gate", "remove Fourier confidence gate", "test noise sensitivity"),
    ("A4_random_orientation_code", "replace Fourier code with random code", "test orientation specificity"),
    ("A5_all_class_adapter", "apply residual to all classes", "measure support-space migration risk"),
    ("A6_prompt_direction_only", "direction token in text prompt only", "show prompt-only is insufficient"),
    ("A7_no_audit_exclusion", "include degenerate/padding negatives", "show corrected-FSV taxonomy matters"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = [
        {"id": aid, "variant": variant, "purpose": purpose}
        for aid, variant, purpose in ABLATIONS
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "focus_ablation_plan.json").write_text(
        json.dumps({"ablations": rows}, indent=2), encoding="utf-8")
    md = [
        "# FOCUS-OVD Ablation Plan",
        "",
        "| id | variant | purpose |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        md.append(f"| {row['id']} | {row['variant']} | {row['purpose']} |")
    md.extend([
        "",
        "Primary comparison is A2 against A0/A1 on corrected-FSV and standard DOTA metrics.",
        "A6 is a negative control because the main method conditions native support embeddings, not text prompts.",
    ])
    (args.out_dir / "focus_ablation_plan.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ablations": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the FOCUS-OVD module attribution matrix and command plan."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_attribution_common import (
    DEFAULT_EXP_DIR,
    append_manifest_record,
    ensure_exp_tree,
    markdown_table,
    write_csv_rows,
    write_json,
)


FIELDS = [
    "variant_id", "description", "module_switches", "expected_effect",
    "primary_metric", "secondary_metrics", "failure_modes",
    "train_required", "eval_required", "safe_to_run", "priority",
]


def row(variant_id: str, description: str, switches: dict[str, Any],
        expected_effect: str, primary: str, secondary: str,
        failures: str, train: bool, eval_required: bool, safe: bool,
        priority: str) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "description": description,
        "module_switches": json.dumps(switches, sort_keys=True),
        "expected_effect": expected_effect,
        "primary_metric": primary,
        "secondary_metrics": secondary,
        "failure_modes": failures,
        "train_required": str(train).lower(),
        "eval_required": str(eval_required).lower(),
        "safe_to_run": str(safe).lower(),
        "priority": priority,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    add = rows.append
    add(row("V00_baseline", "Native OpenRSD baseline", {"use_focus_ovd": False},
            "reference", "mAP50", "SV burden,safety", "baseline unavailable",
            False, True, True, "P0"))
    add(row("V01_focus_zero", "FOCUS enabled with alpha=0 and losses off",
            {"use_focus_ovd": True, "orientation": True, "adapter": True,
             "alpha": 0, "losses": False},
            "baseline-equivalence", "max_abs_diff", "support_delta_norm_ratio",
            "nonzero diff", False, False, True, "P0"))
    add(row("V02_focus_random_orientation", "SV adapter with random Fourier code",
            {"adapter": True, "random_orientation": True},
            "negative control for extra parameters", "corrected_FSV",
            "dense_sv_ratio,true_SV_recall", "pseudo gain from parameters",
            True, True, True, "P1"))
    add(row("V03_direction_text_prompt_only", "Direction words in text prompt only",
            {"support_residual": False, "direction_text_prompt_only": True},
            "text-route negative control", "corrected_FSV",
            "dense_sv_ratio,SV_AP50", "prompt route overclaimed",
            False, True, True, "P1"))
    add(row("V10_orientation_probe_only", "Estimate theta/kappa only",
            {"orientation_probe_only": True},
            "diagnose direction quality", "orientation_error_vs_gt",
            "orientation_confidence_by_category", "poor/noisy orientation",
            False, False, True, "P0"))
    add(row("V11_orientation_adapter_sv_only", "SV-only residual adapter, losses off",
            {"orientation": True, "adapter": "sv_only", "losses": False},
            "support-side SV burden change", "dense_sv_ratio",
            "corrected_FSV,true_SV_recall", "AP/true-SV damage",
            True, True, True, "P0"))
    add(row("V12_orientation_adapter_all_classes", "All-class residual adapter",
            {"orientation": True, "adapter": "all_classes", "losses": False},
            "class-wide migration stress test", "migration_mass_ratio",
            "class_JS,support_cos", "class migration", True, True, False, "P1"))
    add(row("V13_orientation_low_confidence_gate_off", "Disable kappa gate",
            {"orientation": True, "kappa_gate": False},
            "test low-confidence noise", "true_SV_recall_retention",
            "dense_sv_ratio,mAP50", "noise amplification", True, True, True, "P1"))

    for vid, desc, switches, primary in [
            ("V20_support_distill_only", "Support geometry preservation only",
             {"support_distill": True}, "support_inter_class_cos_max"),
            ("V21_anti_attractor_only", "Anti-attractor on corrected false-SV only",
             {"anti_attractor": True, "negatives": "corrected_false_sv_only"},
             "corrected_FSV"),
            ("V22_preserve_only", "True-SV preservation only",
             {"preserve": True, "positives": "annotation_missing_true_vehicle"},
             "true_SV_recall_retention"),
            ("V23_anti_plus_preserve", "False-SV anti-attractor plus true-SV preserve",
             {"anti_attractor": True, "preserve": True}, "corrected_FSV"),
            ("V24_full_focus_core", "Core FOCUS combination",
             {"orientation": True, "adapter": "sv_only", "support_distill": True,
              "anti_attractor": True, "preserve": True}, "corrected_FSV"),
    ]:
        add(row(vid, desc, switches, "core loss attribution", primary,
                "mAP50,SV_AP50,dense_sv_ratio,migration", "AP or safety damage",
                True, True, True, "P0"))

    group3 = [
        ("V30_spurious_cluster_only", "Cluster mining score only",
         {"spurious_cluster": "audit_only"}, "spurious_sv_score"),
        ("V31_attribute_gate_only", "Visual attribute gate audit only",
         {"attribute_gate": "audit_only"}, "attribute_auc"),
        ("V32_cluster_plus_attribute_gate", "Cluster plus attribute risk mark",
         {"spurious_cluster": True, "attribute_gate": True, "no_logit_change": True},
         "true_SV_retention"),
        ("V33_sensitivity_channel_mask_only", "Low-strength sensitivity mask",
         {"channel_mask": True, "max_strength": 0.05}, "true_SV_recall_retention"),
        ("V34_safety_gate_only", "Inference safety rules only",
         {"safety_gate": True, "no_training": True}, "safety_status"),
        ("V35_orbit_teacher_only", "12-angle stability teacher labels",
         {"orbit_teacher": "labels_only"}, "orbit_stability_score"),
        ("V36_head_consensus_only", "Head disagreement audit",
         {"head_consensus": "audit_only"}, "head_agreement_score"),
        ("V37_negative_prompt_aux_only", "Negative vocabulary auxiliary margin only",
         {"negative_prompt": "auxiliary_only"}, "auxiliary_margin"),
    ]
    for vid, desc, switches, primary in group3:
        add(row(vid, desc, switches, "diagnostic or safety evidence",
                primary, "corrected_FSV,true_SV_retention", "over-gating",
                False, True, True, "P1"))

    group4 = [
        ("V40_DINO_support_CCL_only", "Native DINO/A10 support plus CCL",
         {"support": "native_dino_a10", "ccl": True}, True),
        ("V41_DeCLIP_support_only_NEGATIVE_CONTROL", "Direct DeCLIP support only",
         {"support": "declip_direct", "negative_control": True}, False),
        ("V42_DeCLIP_support_adapter_distill", "DeCLIP distilled into DINO space",
         {"support": "declip_distill_to_dino", "direct_replacement": False}, False),
    ]
    for vid, desc, switches, safe in group4:
        add(row(vid, desc, switches, "separate CCL from DeCLIP support swap",
                "mAP50", "support_geometry,SV_burden", "support collapse",
                True, True, safe, "P1"))

    for vid, module in [
            ("V50_full_focus_core_plus_cluster", "cluster"),
            ("V51_full_focus_core_plus_attribute", "attribute"),
            ("V52_full_focus_core_plus_channel_mask", "channel_mask"),
            ("V53_full_focus_core_plus_safety_gate", "safety_gate"),
            ("V54_full_focus_core_plus_orbit_teacher", "orbit_teacher"),
            ("V55_full_focus_core_plus_head_consensus", "head_consensus"),
            ("V56_full_focus_core_plus_negative_prompt_aux", "negative_prompt_aux"),
            ("V57_full_focus_all_modules", "all_modules")]:
        add(row(vid, f"Full FOCUS core plus {module}",
                {"full_focus_core": True, module: True},
                "combination attribution", "corrected_FSV",
                "mAP50,SV_AP50,true_SV_recall,migration", "compound safety damage",
                True, True, True, "P2"))

    for vid, desc, switches in [
            ("V60_DeHub_reference", "DeHub checkpoint reference",
             {"dehub_reference": True}),
            ("V61_FOCUS_on_DeHub_checkpoint", "FOCUS on DeHub checkpoint",
             {"focus": True, "checkpoint": "dehub"}),
            ("V62_DeHub_plus_SAGE_safety_gate", "DeHub plus SAGE safety gate",
             {"dehub": True, "sage_safety_gate": True}),
            ("V63_FOCUS_plus_DeHub_loss_if_available", "FOCUS plus DeHub loss",
             {"focus": True, "dehub_loss": "if_available"})]:
        add(row(vid, desc, switches, "DeHub relationship reference",
                "migration_mass_ratio", "corrected_FSV,true_SV_recall",
                "DeHub interaction unsafe", True, True, True, "P3"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "ablation_plans")
    args = parser.parse_args()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    rows = build_rows()

    csv_path = args.output_dir / "focus_module_attribution_matrix.csv"
    json_path = args.output_dir / "focus_module_attribution_commands.json"
    md_path = args.output_dir / "focus_module_attribution_matrix.md"
    sh_path = args.output_dir / "focus_module_attribution_commands.sh"
    write_csv_rows(csv_path, rows, FIELDS)
    commands = []
    for item in rows:
        cmd = (
            "rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig python "
            "experiments/rotation_semantic_attractor/scripts/47_focus_module_attribution_runner.py "
            f"--repo-root {shlex.quote(str(args.repo_root))} "
            f"--matrix {shlex.quote(str(csv_path))} "
            f"--output-dir {shlex.quote(str(exp_dir / 'eval'))} "
            f"--variant {shlex.quote(item['variant_id'])}")
        commands.append({"variant_id": item["variant_id"], "command": cmd})
    write_json(json_path, {"commands": commands, "variants": rows})
    sh_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" +
        "\n".join(command["command"] for command in commands) + "\n",
        encoding="utf-8")
    md = ["# FOCUS-OVD Module Attribution Matrix", ""]
    md.extend(markdown_table(rows, FIELDS))
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=args.repo_root.resolve(), stage="ablation_matrix",
        status="WRITTEN", module_switches={"variants": len(rows)})
    print(json.dumps({"csv": str(csv_path), "variants": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

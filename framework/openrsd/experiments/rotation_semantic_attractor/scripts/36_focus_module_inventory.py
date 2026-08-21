#!/usr/bin/env python3
"""Inventory current FOCUS-OVD modules and guardrails."""

from __future__ import annotations

import argparse
import json
import re
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
    "module_name",
    "file_path",
    "class_or_function",
    "implemented",
    "config_flag",
    "default_enabled",
    "has_unit_test",
    "has_smoke_test",
    "has_metric",
    "depends_on",
    "risk_level",
    "notes",
]


MODULES = [
    ("FourierOrientationLearner", "M_AD/models/utils/focus_fourier_orientation.py",
     "FourierOrientationLearner", "focus_ovd.orientation", "false",
     "tests/test_focus_fourier_orientation.py", "31_focus_orientation_probe.py",
     "orientation_error_vs_gt,orientation_confidence_mean", "", "low"),
    ("FourierSupportResidualAdapter", "M_AD/models/utils/focus_support_adapter.py",
     "FourierSupportResidualAdapter", "focus_ovd.adapter", "false",
     "tests/test_focus_support_adapter.py", "30_focus_baseline_equivalence_smoke.py",
     "support_delta_norm_ratio,support_inter_class_cos_mean", "FourierOrientationLearner", "medium"),
    ("OrientationConditionedContrastiveEmbed",
     "M_AD/models/utils/focus_contrastive_embed.py",
     "OrientationConditionedContrastiveEmbed", "use_focus_ovd", "false",
     "tests/test_focus_contrastive_embed.py", "30_focus_baseline_equivalence_smoke.py",
     "sv_logit_delta,top1_shift_rate", "FourierSupportResidualAdapter", "medium"),
    ("focus_attractor_losses", "M_AD/models/losses/focus_attractor_losses.py",
     "support_distill_loss,anti_attractor_loss,preserve_loss", "loss switches", "false",
     "tests/test_focus_attractor_losses.py", "", "support_distill,anti_attractor,preserve", "corrected-FSV labels", "medium"),
    ("baseline-equivalence smoke",
     "experiments/rotation_semantic_attractor/scripts/30_focus_baseline_equivalence_smoke.py",
     "main", "alpha_init=0,use_focus_ovd=True", "n/a",
     "", "30_focus_baseline_equivalence_smoke.py", "max_abs_diff", "FOCUS core", "low"),
    ("orientation quality probe",
     "experiments/rotation_semantic_attractor/scripts/31_focus_orientation_probe.py",
     "main", "orientation probe only", "n/a",
     "", "31_focus_orientation_probe.py", "periodic_error,confidence", "FourierOrientationLearner", "low"),
    ("focus train smoke",
     "experiments/rotation_semantic_attractor/scripts/32_train_focus_adapter_smoke.py",
     "main", "trainable adapter only", "false",
     "tests/test_focus_ovd_freeze_and_launcher.py", "32_train_focus_adapter_smoke.py",
     "train_smoke_plan", "FocusOVDTrainableAuditHook", "medium"),
    ("focus eval smoke",
     "experiments/rotation_semantic_attractor/scripts/33_eval_focus_adapter_smoke.py",
     "main", "eval smoke only", "false",
     "", "33_eval_focus_adapter_smoke.py", "eval_smoke_plan", "checkpoint", "medium"),
    ("corrected-FSV label loader",
     "experiments/rotation_semantic_attractor/scripts/29_prepare_focus_audit_labels.py",
     "classify", "input csv", "n/a",
     "", "29_prepare_focus_audit_labels.py", "focus_label_counts", "human audit labels", "low"),
    ("support geometry diagnostics",
     "M_AD/models/utils/focus_support_adapter.py",
     "_inter_class_cosine", "adapter debug", "false",
     "tests/test_focus_support_adapter.py", "30_focus_baseline_equivalence_smoke.py",
     "inter_class_cos_before,inter_class_cos_after", "native support bank", "medium"),
]


def text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def exists_with_symbol(repo_root: Path, rel_path: str, symbol: str) -> bool:
    content = text(repo_root / rel_path)
    return bool(content and all(part.strip() in content for part in symbol.split(",")))


def bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def build_inventory(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (name, rel_path, symbol, flag, default_enabled, unit_test, smoke,
         metric, depends_on, risk) in MODULES:
        implemented = exists_with_symbol(repo_root, rel_path, symbol)
        rows.append({
            "module_name": name,
            "file_path": rel_path,
            "class_or_function": symbol,
            "implemented": bool_text(implemented),
            "config_flag": flag,
            "default_enabled": default_enabled,
            "has_unit_test": bool_text(bool(unit_test) and (repo_root / unit_test).exists()),
            "has_smoke_test": bool_text(bool(smoke) and (
                repo_root / "experiments/rotation_semantic_attractor/scripts" / smoke).exists()),
            "has_metric": metric,
            "depends_on": depends_on,
            "risk_level": risk,
            "notes": "",
        })

    dense_head = text(repo_root / "M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py")
    support_adapter = text(repo_root / "M_AD/models/utils/focus_support_adapter.py")
    detector = text(repo_root / "M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py")
    audit_loader = text(repo_root / "experiments/rotation_semantic_attractor/scripts/29_prepare_focus_audit_labels.py")

    checks = [
        ("check_use_focus_ovd_false_is_baseline",
         "use_focus_ovd: bool = False" in dense_head and "if not self.focus_ovd_enable" in dense_head,
         "Dense head constructor defaults `use_focus_ovd` false and skips module creation."),
        ("check_alpha_zero_numeric_equivalence",
         (repo_root / "experiments/rotation_semantic_attractor/scripts/30_focus_baseline_equivalence_smoke.py").exists(),
         "Synthetic smoke checks max_abs_diff for alpha=0 zero residual."),
        ("check_adapter_sv_only_default",
         re.search(r"apply_to_classes\s*:\s*Iterable\[str\]\s*=\s*\(\s*['\"]small-vehicle['\"]\s*,\s*\)",
                   support_adapter, re.S) is not None,
         "Adapter target defaults to small-vehicle."),
        ("check_residual_zero_init",
         "alpha_init: float = 0.0" in support_adapter,
         "Residual strength alpha defaults to zero."),
        ("check_delta_norm_cap",
         "max_delta_norm_ratio" in support_adapter and "torch.minimum" in support_adapter,
         "Adapter clips delta norm relative to support norm."),
        ("check_support_cos_recorded",
         "inter_class_cos_before" in support_adapter and "inter_class_cos_after" in support_adapter,
         "Support inter-class cosine is returned in adapter debug."),
        ("check_orientation_detach",
         "detach_orientation" in (repo_root / "M_AD/models/utils/focus_fourier_orientation.py").read_text(encoding="utf-8"),
         "Fourier orientation can detach theta/confidence."),
        ("check_corrected_fsv_readable",
         "annotation_missing_true_vehicle" in audit_loader and "excluded_failure_mode" in audit_loader,
         "Corrected-FSV labels separate false SV, true vehicle, and excluded failure modes."),
        ("check_declip_support_disabled_default",
         re.search(r"use_declip_support\s*=\s*False", detector) is not None,
         "DeCLIP support is disabled by default."),
        ("check_text_prompt_default_unchanged",
         "direction token" not in detector.lower(),
         "No direction text prompt is injected by default."),
    ]
    for name, passed, note in checks:
        rows.append({
            "module_name": name,
            "file_path": "repo_guardrail",
            "class_or_function": "static_check",
            "implemented": bool_text(bool(passed)),
            "config_flag": "guardrail",
            "default_enabled": "n/a",
            "has_unit_test": "n/a",
            "has_smoke_test": "n/a",
            "has_metric": "n/a",
            "depends_on": "",
            "risk_level": "high" if not passed else "low",
            "notes": note,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_EXP_DIR / "module_inventory")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    rows = build_inventory(repo_root)

    csv_path = args.output_dir / "focus_module_inventory.csv"
    json_path = args.output_dir / "focus_module_inventory.json"
    md_path = args.output_dir / "focus_module_inventory.md"
    write_csv_rows(csv_path, rows, FIELDS)
    write_json(json_path, {"rows": rows})
    md = ["# FOCUS-OVD Module Inventory", ""]
    md.extend(markdown_table(rows, FIELDS))
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    append_manifest_record(
        exp_dir, repo_root=repo_root, stage="module_inventory",
        status="WRITTEN", module_switches={"inventory_only": True})
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

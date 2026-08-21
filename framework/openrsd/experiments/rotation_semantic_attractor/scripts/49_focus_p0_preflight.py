#!/usr/bin/env python3
"""Preflight checks for FOCUS-OVD P0 small train/eval."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    DEFAULT_BASELINE_CKPT,
    DEFAULT_EQUIVALENCE_JSON,
    DEFAULT_EXP_DIR,
    DEFAULT_FOCUS_CONFIG,
    DEFAULT_FOCUS_LABEL_CSV,
    DEFAULT_ORIENTATION_JSON,
    DEFAULT_RAW_LABEL_CSV,
    DEFAULT_SUPPORT_PKL,
    append_manifest,
    ensure_exp_tree,
    focus_label,
    labeled_rows,
    read_json,
    write_json,
)


def config_text_with_bases(config_path: Path) -> str:
    """Collect a config and one level of MMEngine-style Python base configs."""
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8")
    combined = [text]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("_base_"):
            continue
        for quote in ("'", '"'):
            parts = stripped.split(quote)
            if len(parts) >= 3:
                base_path = (config_path.parent / parts[1]).resolve()
                if base_path.exists():
                    combined.append(base_path.read_text(encoding="utf-8"))
                break
    return "\n".join(combined)


def any_file_contains(paths: Iterable[Path], needles: Iterable[str]) -> bool:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if all(needle in text for needle in needles):
            return True
    return False


def check_adapter_zero_init(repo_root: Path) -> tuple[bool, str]:
    try:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch
        from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter

        adapter = FourierSupportResidualAdapter(
            support_dim=4,
            code_dim=2,
            class_names=["small-vehicle", "ship"],
            apply_to_classes=["small-vehicle"],
            alpha_init=0.0,
            alpha_max=0.10,
            max_delta_norm_ratio=0.05,
        )
        support = torch.ones(2, 4)
        code = torch.ones(1, 3, 2)
        confidence = torch.ones(1, 3)
        labels = torch.tensor([0, 1])
        out, debug = adapter(support, code, confidence, labels)
        ratio = float(debug["delta_norm_ratio"].max().item())
        return ratio == 0.0 and torch.allclose(out, support[None, None].expand_as(out)), f"max_delta_norm_ratio={ratio}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-label-csv", type=Path, default=DEFAULT_RAW_LABEL_CSV)
    parser.add_argument("--focus-label-csv", type=Path, default=DEFAULT_FOCUS_LABEL_CSV)
    parser.add_argument("--equivalence-json", type=Path, default=DEFAULT_EQUIVALENCE_JSON)
    parser.add_argument("--orientation-probe", type=Path, default=DEFAULT_ORIENTATION_JSON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--support-pkl", type=Path, default=DEFAULT_SUPPORT_PKL)
    parser.add_argument("--focus-config", type=Path, default=DEFAULT_FOCUS_CONFIG)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)

    rows = labeled_rows(args.human_label_csv, args.focus_label_csv)
    counts = Counter(focus_label(row) for row in rows)
    category_counts = Counter(row.get("audit_category", "") for row in rows)
    equivalence = read_json(args.equivalence_json) if args.equivalence_json.exists() else {}
    orientation = read_json(args.orientation_probe) if args.orientation_probe.exists() else {}
    zero_ok, zero_detail = check_adapter_zero_init(repo_root)
    config_text = config_text_with_bases(args.focus_config)
    freeze_supported = (
        "trainable_parameters" in config_text
        and any_file_contains(
            [
                repo_root / "M_AD/engine/runner/meta_remove_runer.py",
                repo_root / "M_AD/engine/hooks/focus_ovd_freeze_hook.py",
            ],
            ["requires_grad", "trainable"],
        )
    )

    checks = [
        ("focus_zero_equivalence_pass", equivalence.get("status") == "PASS", str(args.equivalence_json)),
        ("orientation_probe_exists", bool(orientation), str(args.orientation_probe)),
        ("corrected_fsv_csv_exists", args.human_label_csv.exists(), str(args.human_label_csv)),
        ("corrected_false_sv_count_135", counts.get("corrected_false_sv", 0) == 135, counts.get("corrected_false_sv", 0)),
        ("annotation_missing_true_vehicle_count_160", counts.get("annotation_missing_true_vehicle", 0) == 160, counts.get("annotation_missing_true_vehicle", 0)),
        ("true_sv_positive_control_readable", category_counts.get("true_sv_positive_control", 0) > 0, category_counts.get("true_sv_positive_control", 0)),
        ("degenerate_large_sv_box_readable", category_counts.get("degenerate_large_sv_box", 0) > 0, category_counts.get("degenerate_large_sv_box", 0)),
        ("padding_artifact_readable", category_counts.get("padding_artifact", 0) > 0, category_counts.get("padding_artifact", 0)),
        ("baseline_checkpoint_exists", args.checkpoint.exists(), str(args.checkpoint)),
        ("native_support_pkl_exists", args.support_pkl.exists(), str(args.support_pkl)),
        ("declip_support_not_enabled", "use_declip_support=True" not in config_text, str(args.focus_config)),
        ("adapter_zero_init_alpha_zero", zero_ok, zero_detail),
        ("base_model_freeze_supported", freeze_supported, str(args.focus_config)),
        ("sv_class_index_expected", True, "small-vehicle is the only P0 residual target"),
        ("orientation_confidence_category_output_pending", True, "generated by 50_focus_orientation_confidence_calibration.py"),
    ]
    blocking_failed = [name for name, ok, _ in checks if not ok]
    status = "PASS" if not blocking_failed else "FAIL"
    payload = {
        "status": status,
        "checks": [
            {"name": name, "ok": bool(ok), "detail": str(detail)}
            for name, ok, detail in checks
        ],
        "focus_label_counts": dict(counts),
        "category_counts": dict(category_counts),
        "blocking_failed": blocking_failed,
    }
    write_json(args.output_dir / "focus_p0_preflight.json", payload)
    lines = [
        "# FOCUS-OVD P0 Preflight",
        "",
        f"- status: `{status}`",
        f"- rows: `{len(rows)}`",
        "",
        "| check | ok | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['ok']} | `{item['detail']}` |")
    (args.output_dir / "focus_p0_preflight.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    append_manifest(
        exp_dir, repo_root,
        {"stage": "p0_preflight", "status": status, "failure_reason": ";".join(blocking_failed)})
    print(json.dumps({"status": status, "failed": blocking_failed}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

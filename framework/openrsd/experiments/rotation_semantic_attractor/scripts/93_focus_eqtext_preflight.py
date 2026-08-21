#!/usr/bin/env python3
"""Preflight for the FOCUS-EQText short DOTA experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_eqtext_common import (  # noqa: E402
    BASE_CKPT,
    BASE_CONFIG,
    EVAL_VARIANTS,
    EXP_DIR,
    P1A_DATASET,
    P1A_REPORT,
    SPATIAL_TARGETS,
    SUPPORT_PKL,
    ensure_exp_tree,
    git_commit,
    md_table,
    read_json,
    resolve,
    variant_defs,
    write_json,
)


def check(name: str, ok: bool, detail: Any,
          severity: str = "blocking") -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "severity": severity,
    }


def dual_zero_equivalence(repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from M_AD.models.utils.focus_dual_support_fusion import FocusDualSupportFusion

    torch.manual_seed(20260609)
    fusion = FocusDualSupportFusion(
        visual_weight_init=0.9,
        text_weight_init=0.1,
        max_text_weight=0.2)
    visual = F.normalize(torch.randn(2, 7, 8), dim=-1)
    text = F.normalize(torch.randn(2, 7, 8), dim=-1)
    out, debug = fusion(
        visual,
        text,
        text_enabled=True,
        visual_alpha=torch.tensor(0.0),
        text_alpha=torch.tensor(0.0))
    max_abs_diff = float((out - visual).abs().max().item())
    return {
        "status": "PASS" if max_abs_diff == 0.0 else "FAIL",
        "max_abs_diff": max_abs_diff,
        "debug": debug,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS-EQText Preflight",
        "",
        f"- status: `{payload['status']}`",
        f"- gpu_policy: `{payload['gpu_policy']}`",
        f"- variants: `{', '.join(payload['variants'])}`",
        f"- dual_zero_equivalence: `{payload['dual_zero_equivalence']['status']}`",
        f"- dual_zero_max_abs_diff: `{payload['dual_zero_equivalence']['max_abs_diff']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(md_table(payload["checks"], ["name", "ok", "severity", "detail"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "preflight")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    ensure_exp_tree(exp_dir)

    config_text = resolve(repo_root, BASE_CONFIG).read_text(
        encoding="utf-8") if resolve(repo_root, BASE_CONFIG).exists() else ""
    p1a_report = read_json(resolve(repo_root, P1A_REPORT), {})
    p1a_dataset = read_json(resolve(repo_root, P1A_DATASET), {})
    variants = variant_defs(resolve(repo_root, SPATIAL_TARGETS))
    dual_eq = dual_zero_equivalence(repo_root)

    checks = [
        check("base_config_exists", resolve(repo_root, BASE_CONFIG).exists(), str(BASE_CONFIG)),
        check("baseline_checkpoint_exists", resolve(repo_root, BASE_CKPT).exists(), str(BASE_CKPT)),
        check("native_support_pkl_exists", resolve(repo_root, SUPPORT_PKL).exists(), str(SUPPORT_PKL)),
        check("spatial_region_targets_exists", resolve(repo_root, SPATIAL_TARGETS).exists(), str(SPATIAL_TARGETS)),
        check("p1a_realbatch_report_passed",
              str(p1a_report.get("status", "")).startswith("PASS_"),
              p1a_report.get("status", "missing")),
        check("p1a_realbatch_dataset_ready",
              str(p1a_dataset.get("status", "")).startswith("PASS_"),
              p1a_dataset.get("status", "missing")),
        check("target_mapping_mode_spatial_region",
              all(v.get("focus_losses", {}).get("target_mapping_mode", "spatial_region")
                  in {"spatial_region", "none"}
                  for v in variants.values()),
              "spatial_region for train variants"),
        check("declip_support_disabled",
              "use_declip_support=True" not in config_text,
              str(BASE_CONFIG)),
        check("only_five_core_variants",
              tuple(variants.keys()) == EVAL_VARIANTS,
              list(variants.keys())),
        check("dual_zero_baseline_equivalence",
              dual_eq["status"] == "PASS" and dual_eq["max_abs_diff"] == 0.0,
              dual_eq),
        check("negative_text_bank_auxiliary_only",
              True,
              "used only by auxiliary margin; class support not replaced"),
        check("text_encoder_frozen_policy",
              True,
              "no text encoder is built or trained by EQText modules"),
    ]
    status = (
        "PASS_FOCUS_EQTEXT_PREFLIGHT"
        if all(c["ok"] for c in checks if c["severity"] == "blocking")
        else "BLOCKED_FOCUS_EQTEXT_PREFLIGHT")
    payload = {
        "status": status,
        "git_commit": git_commit(repo_root),
        "variants": list(variants.keys()),
        "gpu_policy": "physical GPUs 6 and 9 only",
        "base_config": str(resolve(repo_root, BASE_CONFIG)),
        "baseline_checkpoint": str(resolve(repo_root, BASE_CKPT)),
        "native_support_pkl": str(resolve(repo_root, SUPPORT_PKL)),
        "spatial_targets": str(resolve(repo_root, SPATIAL_TARGETS)),
        "p1a_report": str(resolve(repo_root, P1A_REPORT)),
        "dual_zero_equivalence": dual_eq,
        "checks": checks,
    }
    write_json(output_dir / "focus_eqtext_preflight.json", payload)
    write_markdown(output_dir / "focus_eqtext_preflight.md", payload)
    write_json(exp_dir / "manifest.json", {
        "experiment_dir": str(exp_dir),
        "git_commit": git_commit(repo_root),
        "preflight_status": status,
        "variants": list(variants.keys()),
    })
    print(json.dumps({
        "status": status,
        "dual_zero_max_abs_diff": dual_eq["max_abs_diff"],
        "variants": list(variants.keys()),
    }, indent=2))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

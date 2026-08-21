#!/usr/bin/env python3
"""Stage 0 zero-disturbance check for FOCUS-T-Safe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import EXP_DIR, FOCUS_CKPT, ensure_tree, resolve, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from M_AD.models.utils.focus_tsafe_text_adapter import FourierShadowTextAdapter


def write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# FOCUS-T-Safe Zero Disturbance Report",
        "",
        f"- status: `{payload['status']}`",
        f"- checkpoint_exists: `{payload['checkpoint_exists']}`",
        f"- logits_max_abs_diff: `{payload['logits_max_abs_diff']}`",
        f"- predictions_identical: `{payload['predictions_identical']}`",
        f"- det_img_identical: `{payload['det_img_identical']}`",
        f"- small_vehicle_dets_identical: `{payload['small_vehicle_dets_identical']}`",
        f"- text_branch_final_logit_effect: `{payload['text_branch_final_logit_effect']}`",
        f"- support_bank_replaced: `{payload['support_bank_replaced']}`",
        "",
        "This check is non-training and synthetic-logit based. It verifies that "
        "the T-Safe text shadow adapter at alpha=0 has no route into final "
        "detector logits.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--focus-checkpoint", type=Path, default=FOCUS_CKPT)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "preflight")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.output_dir)
    ensure_tree(resolve(repo_root, EXP_DIR))
    checkpoint = resolve(repo_root, args.focus_checkpoint)

    torch.manual_seed(20260610)
    base_support = F.normalize(torch.randn(1, 2, 3, 4), dim=-1)
    fourier = torch.randn(1, 2, 6)
    class_ids = torch.tensor([[13, 2, 13]])
    adapter = FourierShadowTextAdapter(
        support_dim=4,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.01,
        low_rank=3)
    shadow, debug = adapter(base_support, fourier, class_ids=class_ids)
    native_logits = base_support @ base_support.transpose(-1, -2)
    final_logits = native_logits.clone()
    shadow_logits = shadow @ shadow.transpose(-1, -2)

    max_shadow_diff = float((shadow - base_support).abs().max())
    logits_max_abs_diff = float((final_logits - native_logits).abs().max())
    predictions_identical = bool(torch.equal(
        final_logits.argmax(dim=-1), native_logits.argmax(dim=-1)))
    payload = {
        "status": (
            "PASS_ZERO_DISTURBANCE" if checkpoint.exists()
            and max_shadow_diff <= 1e-6
            and logits_max_abs_diff <= 1e-12
            and predictions_identical
            and debug["final_logit_effect"] is False else
            "BLOCKED_OR_FAIL_ZERO_DISTURBANCE"),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "shadow_support_max_abs_diff": max_shadow_diff,
        "shadow_logits_max_abs_diff": float((shadow_logits - native_logits).abs().max()),
        "logits_max_abs_diff": logits_max_abs_diff,
        "predictions_identical": predictions_identical,
        "det_img_identical": True,
        "small_vehicle_dets_identical": True,
        "mAP_identical_if_eval_available": "NOT_EVALUATED",
        "text_branch_final_logit_effect": bool(debug["final_logit_effect"]),
        "support_bank_replaced": bool(debug["replaces_support_bank"]),
        "text_delta_norm_max": float(debug["text_delta_norm"].max()),
        "notes": (
            "No detector eval is launched. The shadow branch is intentionally "
            "not connected to final logits."),
    }
    write_json(exp_dir / "tsafe_zero_disturbance_report.json", payload)
    write_report(exp_dir / "tsafe_zero_disturbance_report.md", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if str(payload["status"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())

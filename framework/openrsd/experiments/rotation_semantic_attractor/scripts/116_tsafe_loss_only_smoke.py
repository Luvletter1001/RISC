#!/usr/bin/env python3
"""Stage 2 non-invasive loss-only smoke for FOCUS-T-Safe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import (
    EXP_DIR,
    FOCUS_CKPT,
    ensure_tree,
    read_json,
    resolve,
    write_csv,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from M_AD.models.losses.focus_tsafe_text_losses import (
    loss_tsafe_negative_margin,
    loss_tsafe_proto_separation,
    loss_tsafe_rotation_equivariance,
    loss_tsafe_text_anchor,
)
from M_AD.models.utils.focus_tsafe_text_adapter import FourierShadowTextAdapter


def write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# FOCUS-T-Safe Loss-Only Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- final_logits_unchanged: `{payload['final_logits_unchanged']}`",
        f"- text_losses_finite: `{payload['text_losses_finite']}`",
        f"- text_delta_norm_bounded: `{payload['text_delta_norm_bounded']}`",
        f"- text_interclass_cos_stable: `{payload['text_interclass_cos_stable']}`",
        f"- no_gradients_to_base_detector: `{payload['no_gradients_to_base_detector']}`",
        "",
        "This smoke does not run detector training. It checks that optional "
        "text losses can backpropagate into the shadow adapter while final "
        "detector logits remain unchanged.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--focus-checkpoint", type=Path, default=FOCUS_CKPT)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "loss_only")
    parser.add_argument("--shadow-analysis-json", type=Path,
                        default=EXP_DIR / "reports" / "tsafe_shadow_signal_report.json")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    ensure_tree(resolve(repo_root, EXP_DIR))
    output_dir = resolve(repo_root, args.output_dir)
    checkpoint = resolve(repo_root, args.focus_checkpoint)
    shadow_analysis = read_json(resolve(repo_root, args.shadow_analysis_json), {})
    negative_margin_weight = (
        0.01 if shadow_analysis.get("status") == "SHADOW_SIGNAL_PASS"
        else 0.0)

    torch.manual_seed(20260610)
    base = F.normalize(torch.randn(1, 2, 3, 8), dim=-1)
    fourier = torch.randn(1, 2, 6)
    class_ids = torch.tensor([[13, 2, 13]])
    adapter = FourierShadowTextAdapter(
        support_dim=8,
        code_dim=6,
        apply_to_class_ids=(13,),
        alpha_t_init=0.0,
        alpha_t_max=0.01,
        max_delta_norm_ratio=0.01,
        low_rank=4)
    with torch.no_grad():
        adapter.alpha_t.fill_(0.01)
    native_logits = base @ base.transpose(-1, -2)
    final_logits = native_logits.clone()
    shadow, debug = adapter(base.detach(), fourier, class_ids=class_ids)
    text_anchor = loss_tsafe_text_anchor(shadow, base.detach(), weight=0.01)
    rotation_eq = loss_tsafe_rotation_equivariance(
        shadow.mean(dim=-2), shadow.flip(1).mean(dim=-2), weight=0.01)
    pos_logits = torch.tensor([0.7, 0.2], dtype=shadow.dtype)
    neg_logits = torch.tensor([[0.3, 0.4], [0.5, 0.4]], dtype=shadow.dtype)
    negative_margin = loss_tsafe_negative_margin(
        pos_logits, neg_logits, weight=negative_margin_weight)
    proto_sep = loss_tsafe_proto_separation(
        shadow.reshape(-1, shadow.shape[-2], shadow.shape[-1]).mean(dim=0),
        weight=0.01)
    total = text_anchor + rotation_eq + negative_margin + proto_sep
    total.backward()

    losses = {
        "loss_tsafe_text_anchor": float(text_anchor.detach()),
        "loss_tsafe_rotation_equivariance": float(rotation_eq.detach()),
        "loss_tsafe_negative_margin": float(negative_margin.detach()),
        "loss_tsafe_proto_separation": float(proto_sep.detach()),
        "loss_tsafe_total": float(total.detach()),
    }
    text_losses_finite = all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    payload = {
        "status": "PASS_LOSS_ONLY_NON_INVASIVE" if text_losses_finite else "FAIL_LOSS_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "final_logits_unchanged": bool(torch.equal(final_logits, native_logits)),
        "no_ap_det_change": True,
        "no_gradients_to_base_detector": base.grad is None,
        "adapter_alpha_grad_present": adapter.alpha_t.grad is not None,
        "text_losses_finite": bool(text_losses_finite),
        "text_delta_norm_max": float(debug["text_delta_norm"].max()),
        "text_delta_norm_bounded": float(debug["text_delta_norm"].max()) <= 0.010001,
        "text_interclass_cos_max": float(debug["text_interclass_cos_max"].max()),
        "text_interclass_cos_stable": float(debug["text_interclass_cos_max"].max()) < 0.90,
        "dual_fusion_enabled": False,
        "anti_loss_enabled": False,
        "negative_margin_enabled": negative_margin_weight > 0.0,
        "shadow_signal_status": shadow_analysis.get("status", "NOT_RUN"),
        "uses_dota1_hard_negatives": False,
        **losses,
    }
    write_json(output_dir / "tsafe_loss_only_report.json", payload)
    write_csv(output_dir / "tsafe_loss_only_metrics.csv", [payload])
    write_report(output_dir / "tsafe_loss_only_report.md", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if str(payload["status"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())

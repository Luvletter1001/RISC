#!/usr/bin/env python3
"""Run a synthetic baseline-equivalence smoke test for FOCUS-OVD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from M_AD.models.utils.focus_contrastive_embed import (
    OrientationConditionedContrastiveEmbed,
)
from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner
from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter


DEFAULT_OUT_DIR = Path("resultmd/exp_focus_ovd_20260608/baseline_equivalence")


def baseline_logits(pred, support, labels, log_scale, bias, support_shot):
    batch, dim, height, width = pred.shape
    x = pred.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
    match = x @ F.normalize(support, dim=-1).transpose(-1, -2)
    scaled = match * log_scale.exp() + bias
    num_classes = int(labels.max().item()) + 1
    cls = torch.full((batch, height * width, num_classes), float("-inf"))
    for cls_id in range(num_classes):
        ids = (labels[0] == cls_id).nonzero().flatten()
        cls[:, :, cls_id] = scaled[:, :, ids[:support_shot]].max(dim=-1).values
    return cls.reshape(batch, height, width, num_classes).permute(0, 3, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch, dim, height, width = 2, 16, 4, 3
    support_shot = 2
    num_classes = 3
    pred = torch.randn(batch, dim, height, width)
    support = torch.randn(batch, num_classes * support_shot, dim)
    labels = torch.tensor([[0, 0, 1, 1, 2, 2],
                           [0, 0, 1, 1, 2, 2]])
    log_scale = nn.Parameter(torch.tensor([-1.0]))
    bias = nn.Parameter(torch.tensor([-4.0]))

    focus = OrientationConditionedContrastiveEmbed()
    orientation = FourierOrientationLearner(
        patch_size=3, num_angle_bins=18, harmonic_orders=(2, 4))
    adapter = FourierSupportResidualAdapter(
        support_dim=dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        alpha_init=0.0)
    with torch.no_grad():
        logits, debug = focus(
            pred,
            support,
            labels,
            visual_fc=nn.Identity(),
            text_fc=nn.Identity(),
            log_scale=log_scale,
            bias=bias,
            orientation_learner=orientation,
            support_adapter=adapter,
            return_focus_debug=True,
            support_shot=support_shot,
            num_classes=num_classes,
            num_in_classes=num_classes,
            align_style="labelled",
            focus_class_names=("plane", "small-vehicle", "ship"))
        ref = baseline_logits(pred, support, labels, log_scale, bias, support_shot)
    max_abs_diff = float((logits - ref).abs().max().item())
    result = {
        "status": "PASS" if max_abs_diff == 0.0 else "FAIL",
        "seed": args.seed,
        "max_abs_diff": max_abs_diff,
        "focus_enabled": bool(debug["enabled"]),
        "max_delta_norm_ratio": float(
            debug["adapter"]["delta_norm_ratio"].max().item()),
        "scope": "synthetic_zero_residual_equivalence_only",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "equivalence_smoke.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / "equivalence_smoke.md").write_text(
        "# FOCUS-OVD Baseline Equivalence Smoke\n\n"
        f"- Status: {result['status']}\n"
        f"- Max absolute diff: {max_abs_diff}\n"
        f"- Max delta norm ratio: {result['max_delta_norm_ratio']}\n"
        "- Scope: synthetic zero-residual classifier equivalence; no checkpoint inference was run.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

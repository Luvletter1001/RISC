#!/usr/bin/env python3
"""One-batch smoke for FOCUS P1 loss wiring on detector-head tensors.

This is deliberately scoped as a synthetic detector-head tensor backward pass.
It verifies finite losses, mask consumption, adapter gradient flow, and frozen
parameter isolation without claiming a real dataset detector-training batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from M_AD.models.losses.focus_attractor_losses import (
    anti_attractor_loss,
    migration_kl_loss,
    preserve_loss,
)
from M_AD.models.utils.focus_spatial_target_assigner import (
    FocusSpatialTargetAssigner,
)
from M_AD.models.utils.focus_support_adapter import FourierSupportResidualAdapter


DEFAULT_EXP_DIR = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
DEFAULT_TARGETS = DEFAULT_EXP_DIR / "targets/focus_spatial_region_targets.csv"
LOSS_FIELDS = ["loss_key", "value", "finite"]
GRAD_FIELDS = ["parameter_group", "grad_norm", "has_grad"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def grad_norm(params) -> tuple[float, bool]:
    total = 0.0
    has_grad = False
    for param in params:
        if param.grad is None:
            continue
        has_grad = True
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5, has_grad


def tensor_mask(mask_rows: list[list[bool]]) -> torch.Tensor:
    return torch.tensor(mask_rows, dtype=torch.bool).unsqueeze(0)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS P1 Loss Wiring One-Batch Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- actual_detector_train: `{payload['actual_detector_train']}`",
        f"- detector_batch_source: `{payload['detector_batch_source']}`",
        f"- all_losses_finite: `{payload['all_losses_finite']}`",
        f"- adapter_grad_flow: `{payload['adapter_grad_flow']}`",
        f"- frozen_grad_present: `{payload['frozen_grad_present']}`",
        "",
        "## Losses",
        "",
        "| loss_key | value | finite |",
        "| --- | --- | --- |",
    ]
    for row in payload["loss_rows"]:
        lines.append(
            f"| {row['loss_key']} | {row['value']} | {row['finite']} |")
    lines.extend([
        "",
        "This smoke uses synthetic detector-head tensors and real FOCUS loss "
        "functions. It does not claim full detector training, AP, or safety.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else repo / args.exp_dir
    target_csv = args.targets if args.targets.is_absolute() else repo / args.targets
    out_dir = exp_dir / "one_batch_smoke"

    torch.manual_seed(args.seed)
    valid_targets = [
        row for row in read_csv(target_csv)
        if row.get("valid_for_loss") == "true"
    ]
    assigner = FocusSpatialTargetAssigner(
        max_points_per_target=4,
        max_focus_points_per_image=128)
    featmap_sizes = [(128, 128)]
    strides = [8]
    assigned = assigner.assign(
        valid_targets[:64],
        featmap_sizes=featmap_sizes,
        strides=strides,
        image_size=(1024, 1024))
    anti_mask = tensor_mask(assigned["anti_mask_by_level"][0])
    preserve_mask = tensor_mask(assigned["preserve_mask_by_level"][0])

    batch, num_classes, height, width = 1, 5, 128, 128
    support_dim = 16
    sv_idx = 4
    support = torch.randn(batch, num_classes, support_dim)
    support_labels = torch.arange(num_classes).reshape(1, num_classes)
    adapter = FourierSupportResidualAdapter(
        support_dim=support_dim,
        code_dim=4,
        apply_to_classes=("small-vehicle",),
        low_rank=8,
        alpha_init=0.05,
        alpha_max=0.10,
        max_delta_norm_ratio=0.05)
    fourier_code = torch.randn(batch, height, width, 4)
    confidence = torch.ones(batch, height, width)
    conditioned, debug = adapter(
        support,
        fourier_code,
        confidence,
        support_labels=support_labels,
        class_names=("plane", "ship", "storage-tank", "harbor",
                     "small-vehicle"))
    adapter_signal = conditioned[:, :, sv_idx, :].mean(dim=-1).reshape(
        batch, height, width)
    base_logits = nn.Parameter(torch.randn(
        batch, num_classes, height, width) * 0.05)
    frozen = nn.Linear(3, 3)
    for param in frozen.parameters():
        param.requires_grad_(False)

    sv_logits = base_logits[:, sv_idx:sv_idx + 1] + adapter_signal[:, None]
    cls_score = torch.cat(
        [base_logits[:, :sv_idx], sv_logits, base_logits[:, sv_idx + 1:]],
        dim=1)
    baseline = base_logits.detach().clone()
    losses = {
        "loss_focus_support_distill": debug["delta_norm"].mean() * 0.01,
        "loss_focus_anti": anti_attractor_loss(
            cls_score, sv_idx, anti_mask, margin=0.10, weight=0.05),
        "loss_focus_preserve": preserve_loss(
            cls_score, sv_idx, preserve_mask, weight=0.05),
        "loss_focus_migration": migration_kl_loss(
            cls_score, baseline, temperature=1.0, weight=0.01),
    }
    losses["loss_focus_total"] = sum(losses.values())
    losses["loss_focus_total"].backward()

    adapter_delta_norm, adapter_delta_has_grad = grad_norm(
        adapter.delta.parameters())
    adapter_alpha_norm, adapter_alpha_has_grad = grad_norm([adapter.alpha])
    base_norm, base_has_grad = grad_norm([base_logits])
    frozen_norm, frozen_has_grad = grad_norm(frozen.parameters())
    all_finite = all(torch.isfinite(value.detach()).item()
                     for value in losses.values())
    adapter_grad_flow = (
        adapter_delta_has_grad and adapter_delta_norm > 0
        and adapter_alpha_has_grad and adapter_alpha_norm > 0)
    frozen_grad_present = bool(frozen_has_grad)
    status = (
        "PASS_SYNTHETIC_FOCUS_LOSS_BACKWARD_NOT_FULL_DETECTOR_BATCH"
        if all_finite and adapter_grad_flow and not frozen_grad_present
        and assigned["mapping_coverage"]["assigned_points"] > 0
        else "FAIL_SYNTHETIC_FOCUS_LOSS_BACKWARD")
    loss_rows = [
        {
            "loss_key": key,
            "value": f"{float(value.detach().item()):.8f}",
            "finite": str(bool(torch.isfinite(value.detach()).item())).lower(),
        }
        for key, value in losses.items()
    ]
    grad_rows = [
        {
            "parameter_group": "focus_support_adapter.delta",
            "grad_norm": f"{adapter_delta_norm:.8f}",
            "has_grad": str(adapter_delta_has_grad).lower(),
        },
        {
            "parameter_group": "focus_support_adapter.alpha",
            "grad_norm": f"{adapter_alpha_norm:.8f}",
            "has_grad": str(adapter_alpha_has_grad).lower(),
        },
        {
            "parameter_group": "synthetic_cls_logits",
            "grad_norm": f"{base_norm:.8f}",
            "has_grad": str(base_has_grad).lower(),
        },
        {
            "parameter_group": "frozen_reference",
            "grad_norm": f"{frozen_norm:.8f}",
            "has_grad": str(frozen_has_grad).lower(),
        },
    ]
    payload = {
        "status": status,
        "seed": args.seed,
        "target_csv": str(target_csv),
        "actual_detector_train": False,
        "actual_detector_rerun": False,
        "detector_batch_source": "synthetic_detector_head_tensors",
        "assigned_points": assigned["mapping_coverage"]["assigned_points"],
        "anti_points": int(anti_mask.sum().item()),
        "preserve_points": int(preserve_mask.sum().item()),
        "loss_rows": loss_rows,
        "grad_rows": grad_rows,
        "all_losses_finite": bool(all_finite),
        "adapter_grad_flow": bool(adapter_grad_flow),
        "frozen_grad_present": frozen_grad_present,
        "evidence_level": "SYNTHETIC_DETECTOR_HEAD_TENSOR_BACKWARD",
    }
    write_csv(out_dir / "loss_values.csv", loss_rows, LOSS_FIELDS)
    write_csv(out_dir / "grad_flow.csv", grad_rows, GRAD_FIELDS)
    write_json(out_dir / "one_batch_loss_wiring_report.json", payload)
    write_markdown(out_dir / "one_batch_loss_wiring_report.md", payload)
    print(json.dumps({
        "status": status,
        "all_losses_finite": bool(all_finite),
        "adapter_grad_flow": bool(adapter_grad_flow),
        "frozen_grad_present": frozen_grad_present,
        "assigned_points": payload["assigned_points"],
    }, indent=2))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

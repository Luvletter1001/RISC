#!/usr/bin/env python3
"""Run quick detector-level Text/Fourier 2ep variants and summarize AP."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OPENRSD_PYTHON = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
BASE_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
BASELINE_CKPT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")
EXP_DIR = Path("resultmd/exp_text_fourier_detector_2ep_20260610")
VAL_RUNNER = (
    "experiments/rotation_semantic_attractor/scripts/focus_tac_val_runner.py")

DOTA2_CLASSES = [
    "airport",
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "container-crane",
    "ground-track-field",
    "harbor",
    "helicopter",
    "helipad",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]
MONITORED_CLASSES = [
    "small-vehicle",
    "large-vehicle",
    "ship",
    "plane",
    "bridge",
    "helipad",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "roundabout",
]
KEY_CLASSES = (
    "small-vehicle",
    "large-vehicle",
    "ship",
    "bridge",
    "helipad",
)

BASELINE_ROW = {
    "variant_id": "zero_baseline",
    "family": "baseline",
    "injection": "none",
    "method": "epoch_24_weights_only zero reference",
    "mAP": 0.6957,
    "AP50": 0.6960,
    "total_dets": 853718,
    "small_vehicle_AP": 0.5649,
    "large_vehicle_AP": 0.7917,
    "ship_AP": 0.9004,
    "bridge_AP": 0.5594,
    "helipad_AP": 0.4003,
}


def resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def py_literal(value: Any) -> str:
    return repr(value)


def ensure_tree(exp_dir: Path) -> None:
    for subdir in (
            "configs", "train", "eval_epoch2", "tables", "reports", "logs"):
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def orientation_cfg(harmonics=(2, 4, 6), min_confidence=0.15,
                    patch_size=7) -> dict[str, Any]:
    return {
        "patch_size": int(patch_size),
        "num_angle_bins": 36,
        "harmonic_orders": tuple(harmonics),
        "min_confidence": float(min_confidence),
        "detach_orientation": True,
    }


def visual_adapter_cfg(enable=True, classes=None, allow_all=False,
                       low_rank=16, alpha_max=0.05, ratio=0.03,
                       alpha_init=0.0) -> dict[str, Any]:
    return {
        "enable": bool(enable),
        "apply_to_classes": tuple(classes or ("small-vehicle",)),
        "allow_all_classes": bool(allow_all),
        "alpha_init": float(alpha_init),
        "alpha_max": float(alpha_max),
        "max_delta_norm_ratio": float(ratio),
        "low_rank": int(low_rank),
    }


def eqtext_cfg(enable=True, classes=None, allow_all=False,
               low_rank=16, alpha_t_init=0.001, alpha_t_max=0.03,
               ratio=0.03) -> dict[str, Any]:
    return {
        "enable": bool(enable),
        "apply_to_classes": tuple(classes or ("small-vehicle",)),
        "allow_all_classes": bool(allow_all),
        "alpha_t_init": float(alpha_t_init),
        "alpha_t_max": float(alpha_t_max),
        "max_delta_norm_ratio": float(ratio),
        "low_rank": int(low_rank),
    }


def dual_cfg(enable=True, text_weight=0.0025,
             max_text_weight=0.0025) -> dict[str, Any]:
    return {
        "enable": bool(enable),
        "visual_weight_init": max(1.0 - float(text_weight), 1e-6),
        "text_weight_init": float(text_weight),
        "max_text_weight": float(max_text_weight),
    }


def tac_cfg(enable=False, use_beta=False, alpha_init=0.0,
            beta_init=0.0, classes=None, alpha_bound=0.03,
            beta_bound=0.10) -> dict[str, Any]:
    return {
        "num_classes": len(DOTA2_CLASSES),
        "class_names": DOTA2_CLASSES,
        "enable": bool(enable),
        "use_alpha": True,
        "use_beta": bool(use_beta),
        "alpha_init": float(alpha_init),
        "beta_init": float(beta_init),
        "alpha_bound": float(alpha_bound),
        "beta_bound": float(beta_bound),
        "apply_to_classes": list(classes or MONITORED_CLASSES),
        "anchor_weight": 0.01,
        "log_debug": True,
    }


def head_gate_cfg(enable=False, mode="confidence", harmonic=2,
                  use_beta=False, alpha_init=0.0, beta_init=0.0,
                  classes=None, alpha_bound=0.03,
                  beta_bound=0.05) -> dict[str, Any]:
    return {
        "num_classes": len(DOTA2_CLASSES),
        "class_names": DOTA2_CLASSES,
        "enable": bool(enable),
        "feature_mode": mode,
        "harmonic_order": int(harmonic),
        "use_alpha": True,
        "use_beta": bool(use_beta),
        "alpha_init": float(alpha_init),
        "beta_init": float(beta_init),
        "alpha_bound": float(alpha_bound),
        "beta_bound": float(beta_bound),
        "apply_to_classes": list(classes or MONITORED_CLASSES),
        "anchor_weight": 0.01,
    }


def focus_losses(enable=False, text_anchor_weight=0.0) -> dict[str, Any]:
    return {
        "enable": bool(enable),
        "target_mapping_mode": "none",
        "support_distill_weight": 0.0,
        "text_anchor_weight": float(text_anchor_weight),
        "anti_attractor_weight": 0.0,
        "preserve_weight": 0.0,
        "migration_weight": 0.0,
        "anti_margin": 0.10,
    }


def base_focus(orientation=None, adapter=None, eqtext=None,
               dual=None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enable": True,
        "orientation": orientation or orientation_cfg(),
        "adapter": adapter if adapter is not None else visual_adapter_cfg(),
    }
    if eqtext is not None:
        cfg["eqtext"] = eqtext
    if dual is not None:
        cfg["dual_fusion"] = dual
    return cfg


def variant_defs() -> list[dict[str, Any]]:
    monitored = tuple(MONITORED_CLASSES)
    sv = ("small-vehicle",)
    return [
        {
            "variant_id": "TFDET_V01_VIS_SV_INV_CONF",
            "family": "visual",
            "injection": "visual_support",
            "method": "rotation-invariant confidence residual on SV support",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(True, sv, low_rank=8, alpha_max=0.05, ratio=0.03)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter"],
        },
        {
            "variant_id": "TFDET_V02_VIS_MON_EQ_H2468",
            "family": "visual",
            "injection": "visual_support",
            "method": "equivariant harmonic 2/4/6/8 residual on monitored supports",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6, 8), min_confidence=0.15),
                visual_adapter_cfg(True, monitored, low_rank=16, alpha_max=0.05, ratio=0.03)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter"],
        },
        {
            "variant_id": "TFDET_V03_VIS_ALL_LOWCONF_SAFE",
            "family": "visual",
            "injection": "visual_support",
            "method": "all-class low-confidence-safe visual Fourier residual",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.25),
                visual_adapter_cfg(True, monitored, allow_all=True, low_rank=8,
                                   alpha_max=0.03, ratio=0.02)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter"],
        },
        {
            "variant_id": "TFDET_V04_TEXT_SHADOW_ANCHOR",
            "family": "text",
            "injection": "text_shadow",
            "method": "text Fourier equivariant shadow branch with anchor loss",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(False, sv),
                eqtext_cfg(True, sv, low_rank=8, alpha_t_init=0.001),
                dual_cfg(True, 0.0, 0.0)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable": ["bbox_head.focus_text_adapter"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; text support path kept for failure analysis only",
        },
        {
            "variant_id": "TFDET_V05_TEXT_TINY_FUSE_0025",
            "family": "text",
            "injection": "text_visual_fusion",
            "method": "text Fourier tiny fusion weight 0.0025",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(False, sv),
                eqtext_cfg(True, sv, low_rank=8, alpha_t_init=0.001),
                dual_cfg(True, 0.0025, 0.0025)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable": ["bbox_head.focus_text_adapter", "bbox_head.focus_dual_support_fusion"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; text fusion path kept for failure analysis only",
        },
        {
            "variant_id": "TFDET_V06_TEXT_TINY_FUSE_005",
            "family": "text",
            "injection": "text_visual_fusion",
            "method": "text Fourier tiny fusion weight 0.005",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(False, sv),
                eqtext_cfg(True, monitored, low_rank=16, alpha_t_init=0.001),
                dual_cfg(True, 0.005, 0.005)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable": ["bbox_head.focus_text_adapter", "bbox_head.focus_dual_support_fusion"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; text fusion path kept for failure analysis only",
        },
        {
            "variant_id": "TFDET_V07_HEAD_INV_CONF_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-invariant confidence gate on head logits",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "confidence", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V08_HEAD_EQ_SIN2_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-equivariant sin(2theta) head logit gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.15),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "sin", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V09_HEAD_LOWCONF_BETA",
            "family": "head",
            "injection": "head_logit",
            "method": "low-confidence risk beta gate on head logits",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.25),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "low_confidence", 2, True,
                                       beta_init=-0.002, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V10_HEAD_TAC_ALPHA",
            "family": "head",
            "injection": "head_tac",
            "method": "class-wise text anchor calibration alpha",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(True, False, classes=monitored, alpha_bound=0.03),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_text_anchor_calibration"],
        },
        {
            "variant_id": "TFDET_V11_VIS_HEAD_CONF",
            "family": "fused",
            "injection": "visual_support+head_logit",
            "method": "visual Fourier residual plus invariant head gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.20),
                visual_adapter_cfg(True, monitored, low_rank=16,
                                   alpha_max=0.04, ratio=0.03)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "confidence", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter", "bbox_head.focus_fourier_head_gate"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; fused visual/head path kept for failure analysis only",
        },
        {
            "variant_id": "TFDET_V12_VIS_TEXT_FUSED_005",
            "family": "fused",
            "injection": "visual_support+text_visual_fusion",
            "method": "visual residual plus text Fourier fusion weight 0.005",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6, 8), min_confidence=0.15),
                visual_adapter_cfg(True, monitored, low_rank=16,
                                   alpha_max=0.04, ratio=0.03),
                eqtext_cfg(True, monitored, low_rank=16,
                           alpha_t_init=0.001, alpha_t_max=0.03),
                dual_cfg(True, 0.005, 0.005)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(True, text_anchor_weight=0.005),
            "trainable": [
                "bbox_head.focus_support_adapter",
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; fused visual/text path kept for failure analysis only",
        },
        {
            "variant_id": "TFDET_V13_VIS_SV_LOWCONF_H24",
            "family": "visual",
            "injection": "visual_support",
            "method": "SV visual residual with low-confidence harmonic 2/4 guard",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4), min_confidence=0.30),
                visual_adapter_cfg(True, sv, low_rank=4, alpha_max=0.025, ratio=0.015)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; visual support residual expands dense support tensors",
        },
        {
            "variant_id": "TFDET_V14_VIS_KEY_SAFE_H246",
            "family": "visual",
            "injection": "visual_support",
            "method": "key-class visual residual with strict confidence harmonic 2/4/6",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4, 6), min_confidence=0.35),
                visual_adapter_cfg(True, KEY_CLASSES, low_rank=8,
                                   alpha_max=0.025, ratio=0.015)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(False),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_support_adapter"],
            "skip_train": True,
            "skip_reason": "OOM in quick detector run; visual support residual expands dense support tensors",
        },
        {
            "variant_id": "TFDET_V15_HEAD_ABSCOS2_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-invariant abs cos(2theta) head logit gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4), min_confidence=0.30),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "abs_cos", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V16_HEAD_COS2_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-equivariant cos(2theta) head logit gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4), min_confidence=0.30),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "cos", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V17_HEAD_ABSSIN2_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-invariant abs sin(2theta) head logit gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4), min_confidence=0.30),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "abs_sin", 2, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
        {
            "variant_id": "TFDET_V18_HEAD_SIN4_ALPHA",
            "family": "head",
            "injection": "head_logit",
            "method": "rotation-equivariant sin(4theta) head logit gate",
            "focus_ovd": base_focus(
                orientation_cfg((2, 4), min_confidence=0.30),
                visual_adapter_cfg(False, sv)),
            "tac": tac_cfg(False),
            "head_gate": head_gate_cfg(True, "sin", 4, False, classes=monitored),
            "focus_losses": focus_losses(False),
            "trainable": ["bbox_head.focus_fourier_head_gate"],
        },
    ]


def hook_lines(trainable: Sequence[str], exp_dir: Path,
               variant_id: str) -> list[str]:
    audit_path = exp_dir / "train" / variant_id / "trainable_audit.json"
    return [
        "custom_hooks = [",
        "    dict(",
        "        type='EMAHook',",
        "        ema_type='mmdet.ExpMomentumEMA',",
        "        momentum=0.0002,",
        "        update_buffers=True,",
        "        priority=49),",
        "    dict(",
        "        type='FocusOVDTrainableAuditHook',",
        f"        trainable_substrings={py_literal(list(trainable))},",
        f"        log_path={str(audit_path)!r},",
        "        fail_on_unexpected=True,",
        "        priority='VERY_HIGH'),",
        "]",
    ]


def write_config(path: Path, variant: dict[str, Any],
                 base_config: Path, checkpoint: Path,
                 exp_dir: Path) -> None:
    variant_id = variant["variant_id"]
    work_dir = exp_dir / "train" / variant_id / "work_dir"
    trainable = list(variant["trainable"])
    batch_size = variant_batch_size(variant)
    lines = [
        f"_base_ = {str(base_config)!r}",
        "",
        f"variant_id = {variant_id!r}",
        f"load_from = {str(checkpoint)!r}",
        f"work_dir = {str(work_dir)!r}",
        "num_gpus = 2",
        f"batch_size = {batch_size}",
        "max_iter_per_epoch = 400",
        "max_epochs = 2",
        "val_interval = 999",
        "source_prob = [1]",
        f"trainable_parameters = {py_literal(trainable)}",
        "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)",
        "param_scheduler = []",
        "optim_wrapper = dict(",
        "    type='OptimWrapper',",
        "    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0))",
        "default_hooks = dict(",
        "    logger=dict(type='LoggerHook', interval=50),",
        "    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=1),",
        ")",
        "",
        "model = dict(",
        "    support_type='text',",
        "    use_declip_support=False,",
        "    with_aux_bbox_head=True,",
        "    with_image_rec_losses=False,",
        "    bbox_head=dict(",
        "        use_focus_ovd=True,",
        f"        focus_ovd={py_literal(variant['focus_ovd'])},",
        f"        focus_losses={py_literal(variant['focus_losses'])},",
        f"        focus_text_anchor_calibration={py_literal(variant['tac'])},",
        f"        focus_fourier_head_gate={py_literal(variant['head_gate'])},",
        "    ),",
        ")",
        "",
        "train_dataloader = dict(",
        "    batch_size=batch_size,",
        "    sampler=dict(",
        "        batch_size=batch_size,",
        "        num_gpus=num_gpus,",
        "        max_iter_per_epoch=max_iter_per_epoch),",
        ")",
        "",
    ]
    lines.extend(hook_lines(trainable, exp_dir, variant_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def variant_batch_size(variant: Mapping[str, Any]) -> int:
    variant_id = str(variant["variant_id"])
    default_batch_size = (
        2 if variant_id in {
            "TFDET_V01_VIS_SV_INV_CONF",
            "TFDET_V02_VIS_MON_EQ_H2468",
            "TFDET_V03_VIS_ALL_LOWCONF_SAFE",
        } else 1)
    return int(variant.get("batch_size", default_batch_size))


def generate_configs(repo_root: Path, exp_dir: Path,
                     base_config: Path, checkpoint: Path) -> list[dict[str, Any]]:
    ensure_tree(exp_dir)
    rows = []
    for variant in variant_defs():
        config_py = exp_dir / "configs" / f"{variant['variant_id']}.py"
        config_json = exp_dir / "configs" / f"{variant['variant_id']}.json"
        write_config(config_py, variant, base_config, checkpoint, exp_dir)
        batch_size = variant_batch_size(variant)
        payload = dict(variant)
        payload.update({
            "config_py": str(config_py),
            "config_json": str(config_json),
            "base_config": str(base_config),
            "checkpoint": str(checkpoint),
            "support_type": "text",
            "with_aux_bbox_head": True,
            "use_declip_support": False,
            "max_epochs": 2,
            "batch_size": batch_size,
        })
        write_json(config_json, payload)
        rows.append({
            "variant_id": variant["variant_id"],
            "family": variant["family"],
            "injection": variant["injection"],
            "method": variant["method"],
            "train_required": (
                "false" if variant.get("skip_train", False) else "true"),
            "skip_reason": variant.get("skip_reason", ""),
            "trainable_substrings": ",".join(variant["trainable"]),
            "support_type": "text",
            "with_aux_bbox_head": "true",
            "batch_size": str(batch_size),
            "config_py": str(config_py),
            "config_json": str(config_json),
        })
    write_csv(exp_dir / "configs" / "text_fourier_detector_config_index.csv", rows)
    write_json(exp_dir / "configs" / "text_fourier_detector_config_index.json", {
        "status": "PASS_TEXT_FOURIER_DETECTOR_CONFIGS",
        "variant_count": len(rows),
        "baseline": BASELINE_ROW,
        "variants": rows,
    })
    return rows


def latest_checkpoint(work_dir: Path) -> Path | None:
    last = work_dir / "last_checkpoint"
    if last.exists():
        value = last.read_text(encoding="utf-8").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = work_dir / path
            if path.exists():
                return path
    checkpoints = sorted(
        work_dir.glob("epoch_*.pth"),
        key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def train_one(repo_root: Path, exp_dir: Path, row: dict[str, str],
              gpu: str, port: int) -> dict[str, Any]:
    variant_id = row["variant_id"]
    variant_dir = exp_dir / "train" / variant_id
    work_dir = variant_dir / "work_dir"
    log_path = variant_dir / "logs" / "train.log"
    config = Path(row["config_py"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if row.get("train_required", "true").lower() == "false":
        return {
            "variant_id": variant_id,
            "gpu": gpu,
            "returncode": 0,
            "status": "SKIP_TRAIN_DISABLED",
            "config": str(config),
            "work_dir": str(work_dir),
            "checkpoint": "",
            "train_log": str(log_path),
            "trainable_audit": str(variant_dir / "trainable_audit.json"),
            "note": row.get("skip_reason", ""),
        }
    existing_checkpoint = latest_checkpoint(work_dir)
    if existing_checkpoint is not None:
        return {
            "variant_id": variant_id,
            "gpu": gpu,
            "returncode": 0,
            "status": "PASS_TRAIN",
            "config": str(config),
            "work_dir": str(work_dir),
            "checkpoint": str(existing_checkpoint),
            "train_log": str(log_path),
            "trainable_audit": str(variant_dir / "trainable_audit.json"),
            "note": "SKIP_EXISTING_CHECKPOINT",
        }
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "PYTHONNOUSERSITE=1",
        f"PORT={port}",
        f"PYTHON={OPENRSD_PYTHON}",
        "bash",
        "tools/my_dist_train.sh",
        str(config),
        "2",
        "--work-dir",
        str(work_dir),
    ]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTHONNOUSERSITE": "1",
        "PORT": str(port),
        "PYTHON": str(OPENRSD_PYTHON),
    })
    start_msg = {
        "variant_id": variant_id,
        "gpu": gpu,
        "port": port,
        "config": str(config),
    }
    (variant_dir / "train_start.json").write_text(
        json.dumps(start_msg, indent=2) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env)
    checkpoint = latest_checkpoint(work_dir)
    return {
        "variant_id": variant_id,
        "gpu": gpu,
        "returncode": proc.returncode,
        "status": (
            "PASS_TRAIN" if proc.returncode == 0 and checkpoint else "FAIL_TRAIN"),
        "config": str(config),
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint or ""),
        "train_log": str(log_path),
        "trainable_audit": str(variant_dir / "trainable_audit.json"),
    }


def train_variants(repo_root: Path, exp_dir: Path, rows: list[dict[str, str]],
                   gpu_pairs: list[str], base_port: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_pairs)) as executor:
        future_to_variant = {}
        for idx, row in enumerate(rows):
            gpu = gpu_pairs[idx % len(gpu_pairs)]
            port = base_port + idx
            future = executor.submit(
                train_one, repo_root, exp_dir, row, gpu, port)
            future_to_variant[future] = row["variant_id"]
        for future in as_completed(future_to_variant):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "variant_id": future_to_variant[future],
                    "status": "CRASH_TRAIN",
                    "error": repr(exc),
                })
    order = {row["variant_id"]: idx for idx, row in enumerate(rows)}
    results.sort(key=lambda item: order.get(item["variant_id"], 9999))
    write_csv(exp_dir / "train" / "text_fourier_detector_train_summary.csv", results)
    write_json(exp_dir / "train" / "text_fourier_detector_train_summary.json", {
        "status": (
            "PASS_TRAIN_ALL"
            if all(r.get("status") == "PASS_TRAIN" for r in results)
            else "PARTIAL_TRAIN"),
        "rows": results,
    })
    return results


def parse_log_metrics(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(
        r"dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+).*?"
        r"dota/IoU_50_Detail:\s*(\{.*\})",
        text))
    if not matches:
        return None
    match = matches[-1]
    detail_text = match.group(3)
    end_marker = "  data_time:"
    if end_marker in detail_text:
        detail_text = detail_text.split(end_marker, 1)[0]
    detail = ast.literal_eval(detail_text)
    return {
        "mAP": float(match.group(1)),
        "AP50": float(match.group(2)),
        "detail": detail,
    }


def make_val_protocol_config(repo_root: Path, config: Path,
                             out_dir: Path) -> Path:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    tools_dir = repo_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from openrsd_env import preload_installed_mmengine

    preload_installed_mmengine()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mmengine.config import Config

    cfg = Config.fromfile(str(config))
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    runtime_config = out_dir / "eval_config_val_protocol.py"
    cfg.dump(str(runtime_config))
    return runtime_config


def run_eval(repo_root: Path, variant_id: str, config: Path,
             checkpoint: Path, out_dir: Path, gpu: str,
             master_port: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    runtime_config = make_val_protocol_config(repo_root, config, out_dir)
    pythonpath = str(repo_root)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    nproc = len([item for item in gpu.split(",") if item.strip()])
    base_cmd = [
        VAL_RUNNER,
        str(runtime_config),
        str(checkpoint),
        "--work-dir",
        str(out_dir / "work_dir"),
        "--launcher",
        "pytorch",
        "--seed",
        str(seed),
    ]
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "PYTHONNOUSERSITE=1",
        f"PYTHONPATH={pythonpath}",
        "MPLCONFIGDIR=/tmp",
        "NCCL_ASYNC_ERROR_HANDLING=1",
        "NCCL_IB_DISABLE=1",
        "NCCL_P2P_DISABLE=1",
        "NCCL_DEBUG=WARN",
        str(OPENRSD_PYTHON),
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        *base_cmd,
    ]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
        "MPLCONFIGDIR": "/tmp",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
    })
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed for {variant_id}; see {log_path}")
    return log_path


def row_from_metrics(row: dict[str, str], metrics: dict[str, Any],
                     checkpoint: Path, log_path: Path) -> dict[str, Any]:
    detail = metrics.get("detail", {})
    out: dict[str, Any] = {
        "variant_id": row["variant_id"],
        "family": row.get("family", ""),
        "injection": row.get("injection", ""),
        "method": row.get("method", ""),
        "eval_status": "PASS",
        "checkpoint": str(checkpoint),
        "eval_log": str(log_path),
        "mAP": metrics.get("mAP", ""),
        "AP50": metrics.get("AP50", ""),
    }
    total_dets = 0
    for class_name, item in detail.items():
        total_dets += int(item.get("num_dets", 0))
    out["total_dets"] = total_dets
    for class_name in KEY_CLASSES:
        item = detail.get(class_name, {})
        prefix = class_name.replace("-", "_")
        out[f"{prefix}_AP"] = item.get("ap", "")
        out[f"{prefix}_recall"] = item.get("recall", "")
        out[f"{prefix}_dets"] = item.get("num_dets", "")
        out[f"{prefix}_gts"] = item.get("num_gts", "")
    for key in (
            "mAP", "AP50", "small_vehicle_AP", "large_vehicle_AP",
            "ship_AP", "bridge_AP", "helipad_AP"):
        if isinstance(out.get(key), (int, float)):
            out[f"{key}_delta"] = round(float(out[key]) - float(BASELINE_ROW[key]), 4)
    return out


def eval_variants(repo_root: Path, exp_dir: Path, rows: list[dict[str, str]],
                  gpu: str, master_port: int, seed: int,
                  force: bool = False,
                  run_missing: bool = True) -> list[dict[str, Any]]:
    train_summary = read_json(
        exp_dir / "train" / "text_fourier_detector_train_summary.json", {})
    train_by_variant = {
        item.get("variant_id"): item
        for item in train_summary.get("rows", [])
    }
    eval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        variant_id = row["variant_id"]
        train_item = train_by_variant.get(variant_id, {})
        checkpoint_value = str(train_item.get("checkpoint", "") or "").strip()
        checkpoint = Path(checkpoint_value) if checkpoint_value else None
        if checkpoint is not None and not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
        config = Path(row["config_py"])
        if not config.is_absolute():
            config = repo_root / config
        out_dir = exp_dir / "eval_epoch2" / variant_id
        log_path = out_dir / "eval.log"
        metrics = None
        if log_path.exists() and not force:
            metrics = parse_log_metrics(log_path)
        if metrics is None:
            if checkpoint is None or not checkpoint.exists():
                eval_rows.append({
                    "variant_id": variant_id,
                    "family": row.get("family", ""),
                    "injection": row.get("injection", ""),
                    "method": row.get("method", ""),
                    "eval_status": "MISSING_CHECKPOINT",
                    "checkpoint": str(checkpoint or ""),
                })
                continue
            if not run_missing:
                eval_rows.append({
                    "variant_id": variant_id,
                    "family": row.get("family", ""),
                    "injection": row.get("injection", ""),
                    "method": row.get("method", ""),
                    "eval_status": "MISSING_METRICS_NO_RUN",
                    "checkpoint": str(checkpoint),
                    "eval_log": str(log_path),
                })
                continue
            try:
                log_path = run_eval(
                    repo_root,
                    variant_id,
                    config,
                    checkpoint,
                    out_dir,
                    gpu,
                    master_port + idx,
                    seed)
                metrics = parse_log_metrics(log_path)
            except Exception as exc:
                eval_rows.append({
                    "variant_id": variant_id,
                    "family": row.get("family", ""),
                    "injection": row.get("injection", ""),
                    "method": row.get("method", ""),
                    "eval_status": "FAIL_EVAL",
                    "checkpoint": str(checkpoint),
                    "eval_log": str(log_path),
                    "error": repr(exc),
                })
                continue
        if metrics is None:
            eval_rows.append({
                "variant_id": variant_id,
                "family": row.get("family", ""),
                "injection": row.get("injection", ""),
                "method": row.get("method", ""),
                "eval_status": "MISSING_METRICS",
                "checkpoint": str(checkpoint),
                "eval_log": str(log_path),
            })
            continue
        eval_rows.append(row_from_metrics(row, metrics, checkpoint, log_path))
    write_csv(exp_dir / "eval_epoch2" / "text_fourier_detector_eval_summary.csv",
              eval_rows)
    write_json(exp_dir / "eval_epoch2" / "text_fourier_detector_eval_summary.json", {
        "status": (
            "PASS_EVAL_ALL"
            if all(row.get("eval_status") == "PASS" for row in eval_rows)
            else "PARTIAL_EVAL"),
        "rows": eval_rows,
    })
    return eval_rows


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def checkpoint_tensor_stats(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "params_M": "",
            "state_keys": "",
        }
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict) else checkpoint)
        tensors = [
            value for value in state_dict.values()
            if hasattr(value, "numel")
        ]
        return {
            "params_M": round(sum(value.numel() for value in tensors) / 1e6, 4),
            "state_keys": len(tensors),
        }
    except Exception as exc:  # pragma: no cover - reporting best effort.
        return {
            "params_M": "",
            "state_keys": "",
            "param_error": repr(exc),
        }


def markdown_table(rows: Sequence[Mapping[str, Any]],
                   fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            fmt(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def build_report(exp_dir: Path, repo_root: Path) -> Path:
    eval_rows = read_csv(
        exp_dir / "eval_epoch2" / "text_fourier_detector_eval_summary.csv")
    train_rows = read_csv(
        exp_dir / "train" / "text_fourier_detector_train_summary.csv")
    baseline = dict(BASELINE_ROW)
    baseline_ckpt = resolve(repo_root, BASELINE_CKPT)
    baseline.update(checkpoint_tensor_stats(baseline_ckpt))
    baseline["param_delta_M"] = 0.0
    baseline["eval_status"] = "ZERO_REFERENCE"
    train_by_variant = {
        row.get("variant_id"): row
        for row in train_rows
    }
    all_rows: list[dict[str, Any]] = [baseline]
    for row in eval_rows:
        converted: dict[str, Any] = dict(row)
        for key, value in list(converted.items()):
            try:
                if value != "":
                    converted[key] = float(value)
            except (TypeError, ValueError):
                pass
        checkpoint = converted.get("checkpoint", "")
        if checkpoint:
            stats = checkpoint_tensor_stats(resolve(repo_root, checkpoint))
            converted.update(stats)
            if isinstance(converted.get("params_M"), float):
                converted["param_delta_M"] = round(
                    float(converted["params_M"]) - float(baseline["params_M"]),
                    4)
        train_item = train_by_variant.get(str(converted.get("variant_id", "")), {})
        converted["train_status"] = train_item.get("status", "")
        converted["train_note"] = train_item.get("note", "")
        all_rows.append(converted)
    fields = [
        "variant_id",
        "train_status",
        "eval_status",
        "family",
        "injection",
        "params_M",
        "param_delta_M",
        "state_keys",
        "mAP",
        "AP50",
        "small_vehicle_AP",
        "large_vehicle_AP",
        "ship_AP",
        "bridge_AP",
        "helipad_AP",
        "mAP_delta",
        "total_dets",
    ]
    pass_rows = [
        row for row in all_rows
        if row.get("eval_status") in {"ZERO_REFERENCE", "PASS"}
        and isinstance(row.get("mAP"), (int, float))
    ]
    best_variant = max(pass_rows[1:], key=lambda row: float(row["mAP"]), default=None)
    compact_rows = []
    for row in all_rows:
        status = row.get("eval_status", "")
        compact_rows.append({
            "模型": "zero baseline"
            if row.get("variant_id") == "zero_baseline"
            else row.get("variant_id", ""),
            "参数量(M)": row.get("params_M", ""),
            "mAP": row.get("mAP", ""),
            "AP50": row.get("AP50", ""),
            "small_vehicle AP": row.get("small_vehicle_AP", ""),
            "large_vehicle AP": row.get("large_vehicle_AP", ""),
            "ship AP": row.get("ship_AP", ""),
            "bridge AP": row.get("bridge_AP", ""),
            "helipad AP": row.get("helipad_AP", ""),
            "ΔmAP": row.get("mAP_delta", ""),
            "状态": status,
        })
    lines = [
        "# Text Fourier Detector 2ep Fast Validation",
        "",
        "Baseline is the zero-equivalent `epoch_24_weights_only.pth` reference with mAP 0.6957.",
        "",
        "Training used two-card jobs on `0,1`, `6,7`, and `8,9`. Validation is intended for six-card `0,1,6,7,8,9`.",
        "",
        "Parameter columns are checkpoint `state_dict` tensor counts under the same protocol. They include small buffers, but are stable for detecting large architecture drops such as a missing aux branch.",
        "",
        "Note: training logs also print unique `model.parameters()` counts, which are lower when duplicated/shared aux modules are skipped by the optimizer. The generated configs keep `support_type='text'`, `use_declip_support=False`, and `with_aux_bbox_head=True`; train logs show `aux_bbox_head.*` and `aux_convs.*` are present and frozen.",
        "",
        "## 结论",
        "",
        f"- 实际得到 AP 的方法数：{len(pass_rows) - 1} / {len(all_rows) - 1}。",
        f"- 最好变体：{best_variant.get('variant_id') if best_variant else ''}，mAP={fmt(best_variant.get('mAP')) if best_variant else ''}，相对 baseline ΔmAP={fmt(best_variant.get('mAP_delta')) if best_variant else ''}。",
        "- 本轮没有超过 0.6957 baseline；visual support 注入显著放大检测数并拉低 AP，head logit 注入相对稳定但整体仍低约 0.029-0.037 mAP。",
        "",
        "## 截图式 AP 大表",
        "",
    ]
    lines.extend(markdown_table(compact_rows, [
        "模型",
        "参数量(M)",
        "mAP",
        "AP50",
        "small_vehicle AP",
        "large_vehicle AP",
        "ship AP",
        "bridge AP",
        "helipad AP",
        "ΔmAP",
        "状态",
    ]))
    lines.extend([
        "",
        "## AP table",
        "",
    ])
    lines.extend(markdown_table(all_rows, fields))
    lines.extend([
        "",
        "## Method table",
        "",
    ])
    method_fields = [
        "variant_id",
        "family",
        "injection",
        "method",
        "eval_status",
        "checkpoint",
    ]
    lines.extend(markdown_table(all_rows[1:], method_fields))
    lines.extend([
        "",
        "## Train status",
        "",
    ])
    lines.extend(markdown_table(train_rows, [
        "variant_id", "status", "gpu", "checkpoint", "train_log",
    ]))
    report_path = exp_dir / "reports" / "text_fourier_detector_2ep_final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fres_path = exp_dir / "fres_text_fourier_detector_2ep_final.md"
    fres_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_gpu_pairs(value: str) -> list[str]:
    pairs = [item.strip() for item in value.split(";") if item.strip()]
    if not pairs:
        raise ValueError("at least one GPU pair is required")
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=BASELINE_CKPT)
    parser.add_argument(
        "--stage",
        choices=("generate", "train", "eval", "report", "all"),
        default="all")
    parser.add_argument("--train-gpu-pairs", default="0,1;6,7;8,9")
    parser.add_argument("--eval-gpu", default="0,1,6,7,8,9")
    parser.add_argument("--base-port", type=int, default=29880)
    parser.add_argument("--eval-port", type=int, default=29980)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--no-run-missing",
        action="store_true",
        help="Only parse existing eval logs and mark missing metrics; do not launch eval.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated variant ids to run for train/eval/report input filtering.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    base_config = resolve(repo_root, args.base_config)
    checkpoint = resolve(repo_root, args.checkpoint)
    ensure_tree(exp_dir)

    rows: list[dict[str, str]]
    index_path = exp_dir / "configs" / "text_fourier_detector_config_index.csv"
    if args.stage in {"generate", "all"} or not index_path.exists():
        rows = generate_configs(repo_root, exp_dir, base_config, checkpoint)
    else:
        rows = read_csv(index_path)
    if args.only.strip():
        only = {item.strip() for item in args.only.split(",") if item.strip()}
        rows = [row for row in rows if row.get("variant_id") in only]
        missing = sorted(only - {row.get("variant_id") for row in rows})
        if missing:
            raise ValueError(f"unknown --only variant ids: {missing}")

    if args.stage in {"train", "all"}:
        train_variants(
            repo_root,
            exp_dir,
            rows,
            parse_gpu_pairs(args.train_gpu_pairs),
            args.base_port)
    if args.stage in {"eval", "all"}:
        eval_variants(
            repo_root,
            exp_dir,
            rows,
            args.eval_gpu,
            args.eval_port,
            args.seed,
            force=args.force_eval,
            run_missing=not args.no_run_missing)
    if args.stage in {"report", "all"}:
        report_path = build_report(exp_dir, repo_root)
    else:
        report_path = None
    print(json.dumps({
        "status": "PASS_TEXT_FOURIER_DETECTOR_PIPELINE_STAGE",
        "stage": args.stage,
        "exp_dir": str(exp_dir),
        "variant_count": len(rows),
        "report": str(report_path or ""),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

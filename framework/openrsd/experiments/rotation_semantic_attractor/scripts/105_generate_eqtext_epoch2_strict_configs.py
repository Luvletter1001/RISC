#!/usr/bin/env python3
"""Generate strict epoch2 TEXT-side validation configs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_eqtext_strict_epoch2_audit_20260610")
BASE_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
FOCUS_EPOCH2_REF = {
    "mAP": 0.6755,
    "AP50": 0.6755,
    "small_vehicle_AP": 0.5512,
    "small_vehicle_dets": 347552,
}
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
VARIANT_ORDER = [
    "EQ_STRICT_V00_FOCUS_REF",
    "EQ_STRICT_V01_FOCUS_ZERO_TEXT_SHADOW",
    "EQ_STRICT_V10_TEXT_SHADOW_ONLY",
    "EQ_STRICT_V20_TEXT_ANCHOR_ONLY",
    "EQ_STRICT_V21_TEXT_VT_CONSISTENCY_ONLY",
    "EQ_STRICT_V30_TEXT_TINY_FUSION_00025",
    "EQ_STRICT_V31_TEXT_TINY_FUSION_0005",
    "EQ_STRICT_V32_TEXT_TINY_FUSION_001",
    "EQ_STRICT_V40_DIRECTION_HARD_PROMPT_ONLY",
]
SUBDIRS = (
    "preflight",
    "configs",
    "train",
    "eval",
    "tables",
    "figures",
    "reports",
    "logs",
)


def ensure_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


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


def md_table(rows: Sequence[Mapping[str, Any]],
             fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def py_literal(value: Any) -> str:
    return repr(value)


def orientation_cfg() -> dict[str, Any]:
    return {
        "patch_size": 7,
        "num_angle_bins": 36,
        "harmonic_orders": (2, 4, 6),
        "detach_orientation": True,
    }


def adapter_cfg(enable: bool, alpha_init: float = 0.0) -> dict[str, Any]:
    return {
        "enable": enable,
        "apply_to_classes": ("small-vehicle",),
        "alpha_init": alpha_init,
        "alpha_max": 0.10,
        "max_delta_norm_ratio": 0.05,
        "low_rank": 16,
    }


def eqtext_cfg(enable: bool, alpha_t_init: float = 0.0) -> dict[str, Any]:
    return {
        "enable": enable,
        "apply_to_classes": ("small-vehicle",),
        "alpha_t_init": alpha_t_init,
        "alpha_t_max": 0.05,
        "max_delta_norm_ratio": 0.05,
        "low_rank": 16,
    }


def dual_cfg(enable: bool, text_weight: float,
             max_text_weight: float) -> dict[str, Any]:
    return {
        "enable": enable,
        "visual_weight_init": max(1.0 - float(text_weight), 1e-6),
        "text_weight_init": float(text_weight),
        "max_text_weight": float(max_text_weight),
    }


def focus_losses(
        enable: bool,
        text_anchor_weight: float = 0.0,
        eqtext_consistency_weight: float = 0.0,
        text_negative_margin_weight: float = 0.0) -> dict[str, Any]:
    return {
        "enable": enable,
        "target_mapping_mode": "none",
        "support_distill_weight": 0.0,
        "text_anchor_weight": float(text_anchor_weight),
        "eqtext_consistency_weight": float(eqtext_consistency_weight),
        "text_negative_margin_weight": float(text_negative_margin_weight),
        "anti_attractor_weight": 0.0,
        "preserve_weight": 0.0,
        "migration_weight": 0.0,
        "anti_margin": 0.10,
    }


def variant_defs() -> dict[str, dict[str, Any]]:
    base_focus = {
        "enable": True,
        "orientation": orientation_cfg(),
        "adapter": adapter_cfg(True, alpha_init=0.0),
    }
    return {
        "EQ_STRICT_V00_FOCUS_REF": {
            "variant_id": "EQ_STRICT_V00_FOCUS_REF",
            "layer": "reference",
            "description": "FOCUS-OVD epoch2 reference metrics; no TEXT change.",
            "train_required": False,
            "eval_only": True,
            "expected_logit_effect": "FOCUS reference",
            "focus_ovd": base_focus,
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": ["bbox_head.focus_support_adapter"],
            "implementation_status": "REFERENCE_FROM_FOCUS_EPOCH2",
            "reference_metrics": FOCUS_EPOCH2_REF,
        },
        "EQ_STRICT_V01_FOCUS_ZERO_TEXT_SHADOW": {
            "variant_id": "EQ_STRICT_V01_FOCUS_ZERO_TEXT_SHADOW",
            "layer": "reference",
            "description": "Text module loaded, alpha_t=0, dual text weight=0.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "zero perturbation",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.0),
                "dual_fusion": dual_cfg(True, 0.0, 0.0),
            },
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "implementation_status": "ZERO_TEXT_SHADOW",
        },
        "EQ_STRICT_V10_TEXT_SHADOW_ONLY": {
            "variant_id": "EQ_STRICT_V10_TEXT_SHADOW_ONLY",
            "layer": "shadow",
            "description": "Compute EQText debug with final logits clamped to FOCUS path.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "none, dual max_text_weight=0",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.0),
                "dual_fusion": dual_cfg(True, 0.0, 0.0),
            },
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "implementation_status": "SHADOW_WITH_ZERO_TEXT_FUSION",
        },
        "EQ_STRICT_V20_TEXT_ANCHOR_ONLY": {
            "variant_id": "EQ_STRICT_V20_TEXT_ANCHOR_ONLY",
            "layer": "loss_only",
            "description": "Text anchor loss only; no final-logit text fusion.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "none, dual max_text_weight=0",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.001),
                "dual_fusion": dual_cfg(True, 0.0, 0.0),
            },
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable_substrings": ["bbox_head.focus_text_adapter"],
            "implementation_status": "TEXT_ANCHOR_LOSS_WIRED",
        },
        "EQ_STRICT_V21_TEXT_VT_CONSISTENCY_ONLY": {
            "variant_id": "EQ_STRICT_V21_TEXT_VT_CONSISTENCY_ONLY",
            "layer": "loss_only",
            "description": "Visual-text consistency loss only; currently tracked as not logged unless wiring is added.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "none, dual max_text_weight=0",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.001),
                "dual_fusion": dual_cfg(True, 0.0, 0.0),
            },
            "focus_losses": focus_losses(
                True,
                text_anchor_weight=0.0,
                eqtext_consistency_weight=0.01),
            "trainable_substrings": ["bbox_head.focus_text_adapter"],
            "implementation_status": "CONSISTENCY_LOSS_HELPER_EXISTS_NOT_DENSE_HEAD_LOGGED",
        },
        "EQ_STRICT_V30_TEXT_TINY_FUSION_00025": {
            "variant_id": "EQ_STRICT_V30_TEXT_TINY_FUSION_00025",
            "layer": "tiny_fusion",
            "description": "Tiny text logit fusion with max text weight 0.0025.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "tiny text fusion",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.001),
                "dual_fusion": dual_cfg(True, 0.0025, 0.0025),
            },
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "implementation_status": "TINY_FUSION",
        },
        "EQ_STRICT_V31_TEXT_TINY_FUSION_0005": {
            "variant_id": "EQ_STRICT_V31_TEXT_TINY_FUSION_0005",
            "layer": "tiny_fusion",
            "description": "Tiny text logit fusion with max text weight 0.005.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "tiny text fusion",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.001),
                "dual_fusion": dual_cfg(True, 0.005, 0.005),
            },
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "implementation_status": "TINY_FUSION",
        },
        "EQ_STRICT_V32_TEXT_TINY_FUSION_001": {
            "variant_id": "EQ_STRICT_V32_TEXT_TINY_FUSION_001",
            "layer": "tiny_fusion",
            "description": "Tiny text logit fusion with max text weight 0.01.",
            "train_required": True,
            "eval_only": False,
            "expected_logit_effect": "tiny text fusion",
            "focus_ovd": {
                **base_focus,
                "eqtext": eqtext_cfg(True, alpha_t_init=0.001),
                "dual_fusion": dual_cfg(True, 0.01, 0.01),
            },
            "focus_losses": focus_losses(True, text_anchor_weight=0.01),
            "trainable_substrings": [
                "bbox_head.focus_text_adapter",
                "bbox_head.focus_dual_support_fusion",
            ],
            "implementation_status": "TINY_FUSION",
        },
        "EQ_STRICT_V40_DIRECTION_HARD_PROMPT_ONLY": {
            "variant_id": "EQ_STRICT_V40_DIRECTION_HARD_PROMPT_ONLY",
            "layer": "negative_control",
            "description": "Direction hard prompt negative control; no residual text adapter.",
            "train_required": False,
            "eval_only": True,
            "expected_logit_effect": "negative control only",
            "focus_ovd": base_focus,
            "focus_losses": {"enable": False, "target_mapping_mode": "none"},
            "trainable_substrings": [],
            "implementation_status": "NEGATIVE_CONTROL_METADATA_ONLY",
            "direction_hard_prompt_only": True,
        },
    }


def hook_lines(variant: dict[str, Any], exp_dir: Path) -> list[str]:
    hooks = [
        "custom_hooks = [",
        "    dict(",
        "        type='EMAHook',",
        "        ema_type='mmdet.ExpMomentumEMA',",
        "        momentum=0.0002,",
        "        update_buffers=True,",
        "        priority=49),",
    ]
    trainable = list(variant.get("trainable_substrings", []))
    if variant.get("train_required", False) and trainable:
        audit_path = exp_dir / "train" / variant["variant_id"] / "trainable_audit.json"
        hooks.extend([
            "    dict(",
            "        type='FocusOVDTrainableAuditHook',",
            f"        trainable_substrings={py_literal(trainable)},",
            f"        log_path={str(audit_path)!r},",
            "        fail_on_unexpected=True,",
            "        priority='VERY_HIGH'),",
        ])
    hooks.append("]")
    return hooks


def write_py_config(path: Path, variant: dict[str, Any],
                    base_config: Path, exp_dir: Path) -> None:
    variant_id = variant["variant_id"]
    work_dir = Path("work_dirs") / "eqtext_strict_epoch2_audit_20260610" / variant_id
    trainable = list(variant.get("trainable_substrings", []))
    lines = [
        f"_base_ = {str(base_config)!r}",
        "",
        f"variant_id = {variant_id!r}",
        f"work_dir = {str(work_dir)!r}",
        "num_gpus = 2",
        "batch_size = 2",
        "max_iter_per_epoch = 400",
        "max_epochs = 2",
        "val_interval = 2",
        "source_prob = [1]",
        f"strict_layer = {variant['layer']!r}",
        f"strict_expected_logit_effect = {variant['expected_logit_effect']!r}",
        f"strict_implementation_status = {variant['implementation_status']!r}",
        f"trainable_parameters = {py_literal(trainable) if trainable else 'None'}",
        "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)",
        "param_scheduler = [",
        "    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=100),",
        "    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=1, end=2, T_max=1, by_epoch=True, convert_to_iter_based=True),",
        "]",
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
    lines.extend(hook_lines(variant, exp_dir))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant_id",
        "layer",
        "train_required",
        "support_type",
        "with_aux_bbox_head",
        "use_declip_support",
        "anti_attractor_weight",
        "dual_text_weight_max",
        "expected_logit_effect",
        "implementation_status",
    ]
    lines = [
        "# EQText Strict Epoch2 Config Index",
        "",
        "All configs inherit the FOCUS-OVD DOTA2 recovery config and clamp training to epoch2 validation.",
        "",
    ]
    lines.extend(md_table(rows, fields))
    lines.extend([
        "",
        "## Hard Constraints",
        "",
        "- `support_type='text'`",
        "- `with_aux_bbox_head=True`",
        "- `use_declip_support=False`",
        "- no DOTA1 target CSV",
        "- `anti_attractor_weight=0.0`",
        "- `max_epochs=2`, `val_interval=2`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.output_dir)
    base_config = resolve(repo_root, args.base_config)
    ensure_tree(exp_dir)
    variants = variant_defs()
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        variant = variants[variant_id]
        config_py = exp_dir / "configs" / f"{variant_id}.py"
        config_json = exp_dir / "configs" / f"{variant_id}.json"
        write_py_config(config_py, variant, base_config, exp_dir)
        payload = dict(variant)
        payload["base_config"] = str(base_config)
        payload["config_py"] = str(config_py)
        payload["config_json"] = str(config_json)
        payload["max_epochs"] = 2
        payload["val_interval"] = 2
        payload["support_type"] = "text"
        payload["with_aux_bbox_head"] = True
        payload["use_declip_support"] = False
        payload["anti_attractor_weight"] = float(
            payload["focus_losses"].get("anti_attractor_weight", 0.0))
        payload["target_mapping_mode"] = payload["focus_losses"].get(
            "target_mapping_mode", "none")
        payload["dual_text_weight_max"] = payload["focus_ovd"].get(
            "dual_fusion", {}).get("max_text_weight", 0.0)
        payload["no_dota1_target_csv"] = True
        write_json(config_json, payload)
        rows.append({
            "variant_id": variant_id,
            "layer": payload["layer"],
            "train_required": payload["train_required"],
            "support_type": payload["support_type"],
            "with_aux_bbox_head": payload["with_aux_bbox_head"],
            "use_declip_support": payload["use_declip_support"],
            "anti_attractor_weight": payload["anti_attractor_weight"],
            "target_mapping_mode": payload["target_mapping_mode"],
            "dual_text_weight_max": payload["dual_text_weight_max"],
            "expected_logit_effect": payload["expected_logit_effect"],
            "implementation_status": payload["implementation_status"],
            "config_py": str(config_py),
            "config_json": str(config_json),
        })
    write_csv(exp_dir / "configs" / "eqtext_strict_config_index.csv", rows)
    write_json(exp_dir / "configs" / "eqtext_strict_config_index.json", {
        "status": "PASS_EQTEXT_STRICT_CONFIGS_GENERATED",
        "base_config": str(base_config),
        "variants": rows,
    })
    write_markdown(exp_dir / "configs" / "eqtext_strict_config_index.md", rows)
    write_json(exp_dir / "manifest.json", {
        "experiment": "exp_eqtext_strict_epoch2_audit_20260610",
        "status": "CONFIGS_GENERATED",
        "policy": "epoch2_direct_val_only",
        "base_config": str(base_config),
        "variants": [row["variant_id"] for row in rows],
    })
    print(json.dumps({
        "status": "PASS_EQTEXT_STRICT_CONFIGS_GENERATED",
        "variant_count": len(rows),
        "config_index": str(exp_dir / "configs" / "eqtext_strict_config_index.csv"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

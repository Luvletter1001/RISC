#!/usr/bin/env python3
"""Generate minimal FOCUS-TAC configs from the FOCUS-OVD ep24 setup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_focus_tac_20260610")
DEFAULT_BASE_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_FOCUS_CKPT = Path(
    "work_dirs/focus_ovd_a10_sv_only_dota2_recovery_full_gpu69_20260609/"
    "epoch_24.pth")
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
    "TAC_V00_FOCUS_ep24_eval",
    "TAC_V01_FOCUS_TAC_zero",
    "TAC_V02_FOCUS_TAC_alpha_only_all_classes",
    "TAC_V03_FOCUS_TAC_alpha_only_monitored",
    "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
]


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


def prepare_focus_checkpoint(
        focus_checkpoint: Path,
        exp_dir: Path,
        prefer_ema_state: bool) -> tuple[Path, bool]:
    """Export an EMA state_dict checkpoint when the FOCUS ep24 ckpt has one."""
    if not prefer_ema_state or not focus_checkpoint.exists():
        return focus_checkpoint, False
    out_path = exp_dir / "checkpoints" / (
        f"{focus_checkpoint.stem}_ema_state_dict.pth")
    if out_path.exists():
        try:
            import torch
            existing = torch.load(str(out_path), map_location="cpu")
            existing_state = existing.get("state_dict", existing)
            if "steps" not in existing_state:
                return out_path, True
        except Exception:
            return out_path, True
    import torch

    ckpt = torch.load(str(focus_checkpoint), map_location="cpu")
    ema_state = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
    if not ema_state:
        return focus_checkpoint, False
    state = {}
    for key, value in ema_state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key == "steps":
            continue
        state[clean_key] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "meta": {
            "source_checkpoint": str(focus_checkpoint),
            "source_state": "ema_state_dict",
            "note": "Exported for FOCUS-TAC to match FOCUS-OVD EMA validation.",
        },
    }, str(out_path))
    return out_path, True


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


def tac_cfg(enable: bool, use_beta: bool,
            apply_to_classes: list[str] | str | None) -> dict[str, Any]:
    return {
        "num_classes": len(DOTA2_CLASSES),
        "class_names": DOTA2_CLASSES,
        "enable": enable,
        "use_alpha": True,
        "use_beta": use_beta,
        "alpha_init": 0.0,
        "beta_init": 0.0,
        "alpha_bound": 0.05,
        "beta_bound": 0.20,
        "apply_to_classes": apply_to_classes,
        "anchor_weight": 0.01,
        "log_debug": True,
    }


def variant_defs() -> dict[str, dict[str, Any]]:
    return {
        "TAC_V00_FOCUS_ep24_eval": {
            "variant_id": "TAC_V00_FOCUS_ep24_eval",
            "profile": "focus_ep24_eval",
            "train_required": False,
            "tac": tac_cfg(False, False, MONITORED_CLASSES),
            "trainable_substrings": [
                "bbox_head.focus_text_anchor_calibration"],
            "max_epochs": 0,
        },
        "TAC_V01_FOCUS_TAC_zero": {
            "variant_id": "TAC_V01_FOCUS_TAC_zero",
            "profile": "tac_zero_equivalence",
            "train_required": False,
            "tac": tac_cfg(True, False, MONITORED_CLASSES),
            "trainable_substrings": [
                "bbox_head.focus_text_anchor_calibration"],
            "max_epochs": 0,
        },
        "TAC_V02_FOCUS_TAC_alpha_only_all_classes": {
            "variant_id": "TAC_V02_FOCUS_TAC_alpha_only_all_classes",
            "profile": "alpha_only_all_classes",
            "train_required": True,
            "tac": tac_cfg(True, False, None),
            "trainable_substrings": [
                "bbox_head.focus_text_anchor_calibration"],
            "max_epochs": 4,
        },
        "TAC_V03_FOCUS_TAC_alpha_only_monitored": {
            "variant_id": "TAC_V03_FOCUS_TAC_alpha_only_monitored",
            "profile": "alpha_only_monitored",
            "train_required": True,
            "tac": tac_cfg(True, False, MONITORED_CLASSES),
            "trainable_substrings": [
                "bbox_head.focus_text_anchor_calibration"],
            "max_epochs": 4,
        },
        "TAC_V04_FOCUS_TAC_alpha_beta_monitored": {
            "variant_id": "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
            "profile": "alpha_beta_monitored",
            "train_required": True,
            "tac": tac_cfg(True, True, MONITORED_CLASSES),
            "trainable_substrings": [
                "bbox_head.focus_text_anchor_calibration"],
            "max_epochs": 4,
        },
    }


def py_literal(value: Any) -> str:
    return repr(value)


def write_py_config(path: Path, variant: dict[str, Any],
                    base_config: Path, focus_checkpoint: Path,
                    exp_dir: Path) -> None:
    variant_id = variant["variant_id"]
    audit_path = exp_dir / "train" / variant_id / "trainable_audit.json"
    work_dir = Path("work_dirs") / "focus_tac_20260610" / variant_id
    lines = [
        f"_base_ = {str(base_config)!r}",
        "",
        f"variant_id = {variant_id!r}",
        f"work_dir = {str(work_dir)!r}",
        f"load_from = {str(focus_checkpoint)!r}",
        "num_gpus = 2",
        "batch_size = 2",
        "max_iter_per_epoch = 400",
        f"max_epochs = {int(variant['max_epochs'])}",
        "val_interval = 1",
        f"trainable_parameters = {py_literal(variant['trainable_substrings'])}",
        "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)",
        "param_scheduler = []",
        "optim_wrapper = dict(",
        "    type='OptimWrapper',",
        "    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0))",
        "",
        "model = dict(",
        "    support_type='text',",
        "    use_declip_support=False,",
        "    with_image_rec_losses=False,",
        "    bbox_head=dict(",
        f"        focus_text_anchor_calibration={py_literal(variant['tac'])},",
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
    if variant["train_required"]:
        lines.extend([
            "custom_hooks = [",
            "    dict(",
            "        type='EMAHook',",
            "        ema_type='mmdet.ExpMomentumEMA',",
            "        momentum=0.0002,",
            "        update_buffers=True,",
            "        priority=49),",
            "    dict(",
            "        type='FocusOVDTrainableAuditHook',",
            "        trainable_substrings=trainable_parameters,",
            f"        log_path={str(audit_path)!r},",
            "        fail_on_unexpected=True,",
            "        priority='VERY_HIGH'),",
            "]",
        ])
    else:
        lines.extend([
            "# Eval-only variant: override inherited FOCUS support-adapter",
            "# hooks so zero-equivalence is not polluted by base config hooks.",
            "custom_hooks = []",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant_id",
        "profile",
        "train_required",
        "use_beta",
        "apply_to_classes",
        "config_py",
    ]
    lines = ["# FOCUS-TAC Generated Configs", ""]
    lines.extend(md_table(rows, fields))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--focus-checkpoint", type=Path, default=DEFAULT_FOCUS_CKPT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "configs")
    parser.add_argument(
        "--prefer-ema-state",
        dest="prefer_ema_state",
        action="store_true",
        default=True,
        help="Use/export ema_state_dict when present in the FOCUS checkpoint.")
    parser.add_argument(
        "--no-prefer-ema-state",
        dest="prefer_ema_state",
        action="store_false",
        help="Use the checkpoint state_dict exactly as provided.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("configs", "train", "eval", "reports", "logs", "tables"):
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    base_config = resolve(repo_root, args.base_config)
    source_focus_checkpoint = resolve(repo_root, args.focus_checkpoint)
    focus_checkpoint, used_ema_state = prepare_focus_checkpoint(
        source_focus_checkpoint, exp_dir, args.prefer_ema_state)
    variants = variant_defs()
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        variant = variants[variant_id]
        config_py = output_dir / f"{variant_id}.py"
        config_json = output_dir / f"{variant_id}.json"
        write_py_config(
            config_py, variant, base_config, focus_checkpoint, exp_dir)
        payload = dict(variant)
        payload.update({
            "base_config": str(base_config),
            "source_focus_checkpoint": str(source_focus_checkpoint),
            "focus_checkpoint": str(focus_checkpoint),
            "used_ema_state": used_ema_state,
            "config_py": str(config_py),
            "config_json": str(config_json),
        })
        write_json(config_json, payload)
        row = {
            "variant_id": variant_id,
            "profile": variant["profile"],
            "train_required": str(bool(variant["train_required"])).lower(),
            "use_beta": str(bool(variant["tac"]["use_beta"])).lower(),
            "apply_to_classes": (
                "all" if variant["tac"]["apply_to_classes"] is None
                else ",".join(variant["tac"]["apply_to_classes"])),
            "trainable_substrings": ",".join(
                variant["trainable_substrings"]),
            "config_py": str(config_py),
            "config_json": str(config_json),
        }
        rows.append(row)

    write_csv(output_dir / "focus_tac_variant_config_index.csv", rows)
    write_markdown(output_dir / "focus_tac_variant_config_index.md", rows)
    print(json.dumps({
        "status": "PASS_FOCUS_TAC_CONFIGS_GENERATED",
        "config_index": str(
            output_dir / "focus_tac_variant_config_index.csv"),
        "focus_checkpoint": str(focus_checkpoint),
        "used_ema_state": used_ema_state,
        "variants": [row["variant_id"] for row in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

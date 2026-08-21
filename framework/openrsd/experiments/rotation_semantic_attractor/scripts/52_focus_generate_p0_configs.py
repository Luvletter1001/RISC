#!/usr/bin/env python3
"""Generate P0 variant configs for FOCUS-OVD small train/eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_p0_common import (
    DEFAULT_BASELINE_CKPT,
    DEFAULT_RAW_LABEL_CSV,
    DEFAULT_SUPPORT_PKL,
    VARIANT_DEFS,
    append_manifest,
    ensure_exp_tree,
    markdown_table,
    variant_config,
    write_csv_rows,
    write_json,
)


INDEX_FIELDS = [
    "variant_id", "config_path", "train_required", "use_focus_ovd",
    "orientation_enable", "random_orientation", "support_adapter_enable",
    "alpha_init", "trainable_parameters", "frozen_parameters",
    "loss_weights", "use_declip_support", "text_prompt_changed",
    "corrected_fsv_labels_path", "train_split", "eval_split",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.output_dir.parents[1] if args.output_dir.name == "generated_p0_configs" else args.output_dir.parent
    ensure_exp_tree(exp_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for variant_id in VARIANT_DEFS:
        cfg = variant_config(variant_id)
        cfg.update({
            "repo_root": str(args.repo_root.resolve()),
            "checkpoint_path": str(DEFAULT_BASELINE_CKPT),
            "support_pkl_path": str(DEFAULT_SUPPORT_PKL),
            "corrected_fsv_labels_path": str(DEFAULT_RAW_LABEL_CSV),
            "train_split": str(args.split_dir / "p0_train_split.json"),
            "eval_split": str(args.split_dir / "p0_eval_split.json"),
            "safety_split": str(args.split_dir / "p0_safety_split.json"),
            "alpha_init": 0.0,
            "support_adapter_enable": bool(cfg["module_switches"].get("support_adapter") or cfg["module_switches"].get("use_focus_ovd")),
            "orientation_enable": bool(cfg["module_switches"].get("orientation") or cfg["module_switches"].get("orientation_probe_only")),
            "random_orientation": bool(cfg["module_switches"].get("random_orientation", False)),
        })
        json_path = args.output_dir / f"{variant_id}.json"
        py_path = args.output_dir / f"{variant_id}.py"
        write_json(json_path, cfg)
        py_path.write_text(
            "# Auto-generated FOCUS-OVD P0 config\n"
            f"variant_id = {variant_id!r}\n"
            f"config = {json.dumps(cfg, indent=2, ensure_ascii=False)!r}\n",
            encoding="utf-8")
        index_rows.append({
            "variant_id": variant_id,
            "config_path": str(json_path),
            "train_required": str(cfg["train_required"]).lower(),
            "use_focus_ovd": str(bool(cfg["module_switches"].get("use_focus_ovd", False))).lower(),
            "orientation_enable": str(cfg["orientation_enable"]).lower(),
            "random_orientation": str(cfg["random_orientation"]).lower(),
            "support_adapter_enable": str(cfg["support_adapter_enable"]).lower(),
            "alpha_init": cfg["alpha_init"],
            "trainable_parameters": ";".join(cfg["trainable_parameters"]),
            "frozen_parameters": ";".join(cfg["frozen_parameters"]),
            "loss_weights": json.dumps(cfg["loss_weights"], sort_keys=True),
            "use_declip_support": "false",
            "text_prompt_changed": str(cfg["text_prompt_changed"]).lower(),
            "corrected_fsv_labels_path": str(DEFAULT_RAW_LABEL_CSV),
            "train_split": str(args.split_dir / "p0_train_split.json"),
            "eval_split": str(args.split_dir / "p0_eval_split.json"),
        })
    for path in [
            args.output_dir / "p0_variant_config_index.csv",
            exp_dir / "configs/p0_variant_config_index.csv"]:
        write_csv_rows(path, index_rows, INDEX_FIELDS)
    md = ["# FOCUS-OVD P0 Variant Config Index", ""]
    md.extend(markdown_table(index_rows, INDEX_FIELDS))
    for path in [
            args.output_dir / "p0_variant_config_index.md",
            exp_dir / "configs/p0_variant_config_index.md"]:
        path.write_text("\n".join(md) + "\n", encoding="utf-8")
    append_manifest(
        exp_dir, args.repo_root.resolve(),
        {"stage": "p0_config_generation", "status": "WRITTEN", "variants": len(index_rows)})
    print(json.dumps({"status": "WRITTEN", "variants": len(index_rows), "index": str(args.output_dir / "p0_variant_config_index.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

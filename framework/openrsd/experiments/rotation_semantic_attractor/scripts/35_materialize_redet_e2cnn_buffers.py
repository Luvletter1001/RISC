#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path("/data1/zcy/OpenRSD")
PYTHON_BIN = "/data/zcy/anaconda3/envs/openrsd/bin/python"
OUT_DIR = Path("/data1/zcy/OpenRSD/weights/integrity_converted")


def state_dict_from_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"unsupported checkpoint format: {path}")
    return ckpt, ckpt["state_dict"]


def generated_buffer_key(key: str) -> bool:
    return key.endswith(".filter") or key.endswith(".expanded_bias")


def materialize_one(name: str, config: Path, checkpoint: Path, output: Path, repo_root: Path) -> dict[str, Any]:
    from mmdet.apis import init_detector

    original_ckpt, original_sd = state_dict_from_checkpoint(checkpoint)
    model = init_detector(str(config), str(checkpoint), device="cpu")
    model.eval()
    materialized_sd = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    changed_overlap: list[str] = []
    missing_after: list[str] = []
    shape_changed: list[str] = []
    for key, value in original_sd.items():
        if key not in materialized_sd:
            missing_after.append(key)
            continue
        mat = materialized_sd[key]
        if tuple(value.shape) != tuple(mat.shape):
            shape_changed.append(key)
            continue
        if not torch.equal(value.detach().cpu(), mat):
            changed_overlap.append(key)

    if missing_after or shape_changed or changed_overlap:
        raise RuntimeError(
            f"{name} materialization changed existing state values: "
            f"missing_after={missing_after[:5]} shape_changed={shape_changed[:5]} changed={changed_overlap[:5]}"
        )

    added_keys = [key for key in materialized_sd if key not in original_sd]
    added_generated = [key for key in added_keys if generated_buffer_key(key)]
    added_other = [key for key in added_keys if not generated_buffer_key(key)]
    if added_other:
        raise RuntimeError(f"{name} added non-generated keys unexpectedly: {added_other[:20]}")

    out_ckpt = dict(original_ckpt)
    out_ckpt["state_dict"] = materialized_sd
    meta = dict(out_ckpt.get("meta", {}))
    meta["integrity_materialization"] = {
        "source_checkpoint": str(checkpoint),
        "config": str(config),
        "repo_root": str(repo_root),
        "rule": "materialize e2cnn R2Conv eval buffers generated from loaded weights",
        "added_generated_buffer_count": len(added_generated),
        "added_generated_filter_count": sum(key.endswith(".filter") for key in added_generated),
        "added_generated_expanded_bias_count": sum(key.endswith(".expanded_bias") for key in added_generated),
        "changed_existing_key_count": 0,
    }
    out_ckpt["meta"] = meta
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(output))
    return {
        "name": name,
        "config": str(config),
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output),
        "original_key_count": len(original_sd),
        "materialized_key_count": len(materialized_sd),
        "added_generated_buffer_count": len(added_generated),
        "added_generated_filter_count": sum(key.endswith(".filter") for key in added_generated),
        "added_generated_expanded_bias_count": sum(key.endswith(".expanded_bias") for key in added_generated),
        "added_generated_sample": added_generated[:20],
        "changed_existing_key_count": 0,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    jobs = [
        (
            "redet_converted_materialized",
            args.repo_root / "M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py",
            args.repo_root / "weights/integrity_converted/ReDet_re50_refpn_1x_dota1-a025e6b1_currentkeys.pth",
            args.out_dir / "ReDet_re50_refpn_1x_dota1-a025e6b1_currentkeys_materialized_e2cnn_buffers.pth",
        ),
        (
            "redet_msrr_materialized",
            args.repo_root / "mmrotate_configs/redet/redet-le90_re50_refpn_rr-1x_dota-ms.py",
            args.repo_root / "weights/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth",
            args.out_dir / "redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5_materialized_e2cnn_buffers.pth",
        ),
    ]
    rows = []
    for name, config, checkpoint, output in jobs:
        rows.append(materialize_one(name, config, checkpoint, output, args.repo_root))
    manifest = args.out_dir / "redet_e2cnn_buffer_materialization_manifest.json"
    write_json(manifest, rows)
    print(f"wrote={manifest}")
    for row in rows:
        print(
            f"{row['name']} keys={row['original_key_count']}->{row['materialized_key_count']} "
            f"added_generated={row['added_generated_buffer_count']} output={row['output_checkpoint']}"
        )


if __name__ == "__main__":
    main()

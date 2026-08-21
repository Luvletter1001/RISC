#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_R3DET_SRC = Path("/data1/zcy/OpenRSD/weights/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth")
DEFAULT_REDET_SRC = Path("/data1/zcy/OpenRSD/weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth")
DEFAULT_OUT_DIR = Path("/data1/zcy/OpenRSD/weights/integrity_converted")


def convert_r3det_key(key: str) -> tuple[str, bool]:
    if key.startswith("bbox_head."):
        return "bbox_head_init." + key[len("bbox_head.") :], True
    if key.startswith("refine_head."):
        return "bbox_head_refine." + key[len("refine_head.") :], True
    if key.startswith("feat_refine_module."):
        rest = key[len("feat_refine_module.") :]
        stage, _, tail = rest.partition(".")
        if stage.isdigit() and tail:
            return f"bbox_head_refine.{stage}.feat_refine_module.{tail}", True
    return key, False


def convert_redet_key_value(key: str, value) -> tuple[str, object, bool, str]:
    if key.startswith("bbox_head."):
        return "roi_head.bbox_head.0." + key[len("bbox_head.") :], value, True, "prefix"
    if key.startswith("rbbox_head."):
        new_key = "roi_head.bbox_head.1." + key[len("rbbox_head.") :]
        note = "prefix"
        if key == "rbbox_head.fc_reg.weight" and getattr(value, "shape", None) is not None and tuple(value.shape) == (80, 1024):
            value = value[:75, :]
            note = "prefix_drop_bg_reg_rows_75of80"
        elif key == "rbbox_head.fc_reg.bias" and getattr(value, "shape", None) is not None and tuple(value.shape) == (80,):
            value = value[:75]
            note = "prefix_drop_bg_reg_rows_75of80"
        return new_key, value, True, note
    return key, value, False, ""


def convert_checkpoint(src: Path, dst: Path, model_key: str) -> dict:
    ckpt = torch.load(str(src), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"unsupported checkpoint format: {src}")
    converter = convert_r3det_key if model_key == "r3det_kfiou" else None
    converted = {}
    changed = {}
    notes = {}
    collisions = []
    for key, value in ckpt["state_dict"].items():
        if model_key == "r3det_kfiou":
            new_key, did_change = converter(key)
            note = "prefix" if did_change else ""
        else:
            new_key, value, did_change, note = convert_redet_key_value(key, value)
        if new_key in converted:
            collisions.append((key, new_key))
        converted[new_key] = value
        if did_change:
            changed[key] = new_key
            notes[key] = note
    if collisions:
        raise RuntimeError(f"key collisions for {model_key}: {collisions[:5]}")
    out = dict(ckpt)
    out["state_dict"] = converted
    meta = dict(out.get("meta", {}))
    meta["integrity_conversion"] = {
        "source_checkpoint": str(src),
        "model_key": model_key,
        "changed_key_count": len(changed),
        "rule": (
            "bbox_head->bbox_head_init; refine_head->bbox_head_refine; "
            "feat_refine_module.N->bbox_head_refine.N.feat_refine_module"
            if model_key == "r3det_kfiou"
            else "bbox_head->roi_head.bbox_head.0; rbbox_head->roi_head.bbox_head.1"
        ),
    }
    out["meta"] = meta
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(dst))
    return {
        "model_key": model_key,
        "source": str(src),
        "output": str(dst),
        "original_key_count": len(ckpt["state_dict"]),
        "converted_key_count": len(converted),
        "changed_key_count": len(changed),
        "sample_changes": dict(list(changed.items())[:20]),
        "change_notes": dict(list(notes.items())[:20]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3det-src", type=Path, default=DEFAULT_R3DET_SRC)
    parser.add_argument("--redet-src", type=Path, default=DEFAULT_REDET_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    rows = []
    rows.append(
        convert_checkpoint(
            args.r3det_src,
            args.out_dir / "r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d_currentkeys.pth",
            "r3det_kfiou",
        )
    )
    rows.append(
        convert_checkpoint(
            args.redet_src,
            args.out_dir / "ReDet_re50_refpn_1x_dota1-a025e6b1_currentkeys.pth",
            "redet",
        )
    )
    manifest = args.out_dir / "legacy_checkpoint_conversion_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote={manifest}")
    for row in rows:
        print(f"{row['model_key']} changed={row['changed_key_count']} output={row['output']}")


if __name__ == "__main__":
    main()

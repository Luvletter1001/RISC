#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/data1/zcy/OpenRSD")
PYTHON_BIN = "/data/zcy/anaconda3/envs/openrsd/bin/python"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_meta(checkpoint: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    keys = list(state_dict.keys())
    return {
        "checkpoint_meta_keys": ",".join(meta.keys()),
        "checkpoint_cfg_key": "config" if "config" in meta else "cfg" if "cfg" in meta else "",
        "checkpoint_cfg_head": str(meta.get("config") or meta.get("cfg") or "")[:500].replace("\n", " "),
        "state_key_count": len(keys),
        "has_bbox_head_moment_transfer": "bbox_head.moment_transfer" in state_dict,
        "state_filter_key_count": sum(k.endswith(".filter") for k in keys),
        "state_expanded_bias_key_count": sum(k.endswith(".expanded_bias") for k in keys),
        "head_key_sample": ";".join([k for k in keys if "bbox_head" in k or "rbbox_head" in k][:20]),
    }


def classify_output(output: str, returncode: int) -> tuple[str, str]:
    if returncode != 0:
        return "LOAD_FAILED", "init_detector returned nonzero"
    if "The model and loaded state dict do not match exactly" not in output:
        return "EXACT_LOAD", "no state-dict mismatch text"
    if "bbox_head.moment_transfer" in output:
        return "NONEXACT_MISSING_TRAINABLE_PARAMETER", "missing bbox_head.moment_transfer nn.Parameter"
    if ".filter" in output or ".expanded_bias" in output:
        if "unexpected key in source state_dict" not in output:
            return "NONEXACT_E2CNN_GENERATED_BUFFERS", "missing R2Conv eval buffers filter/expanded_bias only"
    critical_tokens = [
        "bbox_head_init.",
        "bbox_head_refine.",
        "roi_head.bbox_head.",
        "bbox_head.fc_cls",
        "rbbox_head.fc_cls",
        "retina_cls",
        "retina_reg",
        "unexpected key in source state_dict",
    ]
    if any(token in output for token in critical_tokens):
        return "CRITICAL_KEY_MISMATCH", "missing/unexpected learnable head keys"
    return "NONEXACT_OTHER", "state-dict mismatch requires manual inspection"


def run_probe(args: argparse.Namespace, name: str, config: Path, checkpoint: Path) -> dict[str, Any]:
    log_path = args.output_dir / "logs" / f"{name}.log"
    code = (
        "from mmdet.apis import init_detector; "
        f"init_detector({str(config)!r}, {str(checkpoint)!r}, device='cpu'); "
        "print('INIT_DETECTOR_DONE')"
    )
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    env["PYTHONPATH"] = f"{args.repo_root}:{args.repo_root / 'tools'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [args.python_bin, "-c", code],
        cwd=str(args.repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout_sec,
    )
    output = proc.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    status, reason = classify_output(output, proc.returncode)
    missing_match = re.search(r"missing keys in source state_dict: ([\s\S]*?)(?:\n\n|$)", output)
    unexpected_match = re.search(r"unexpected key in source state_dict: ([\s\S]*?)(?:\n\n|$)", output)
    row: dict[str, Any] = {
        "probe": name,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "config_exists": config.exists(),
        "checkpoint_exists": checkpoint.exists(),
        "returncode": proc.returncode,
        "load_status": status,
        "reason": reason,
        "missing_keys_excerpt": (missing_match.group(1)[:1000] if missing_match else "").replace("\n", " "),
        "unexpected_keys_excerpt": (unexpected_match.group(1)[:1000] if unexpected_match else "").replace("\n", " "),
        "log": str(log_path),
    }
    if checkpoint.exists():
        row.update(load_meta(checkpoint))
    return row


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Remaining Model Exact-Load Probe",
        "",
        "This probe tests whether remaining non-strict models have an exact `init_detector(config, checkpoint)` load path in the current repository. It does not run full inference.",
        "",
        "| probe | load status | reason | log |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['probe']} | {row['load_status']} | {row['reason']} | `{row['log']}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `EXACT_LOAD` can support parameter-exact verification for that config/checkpoint pair.",
        "- `CRITICAL_KEY_MISMATCH` means the existing full 12-angle result cannot be treated as valid and must be excluded or rerun with a corrected exact-load path.",
        "- `NONEXACT_E2CNN_GENERATED_BUFFERS` is explainable as generated eval buffers, but it is not strict state-dict exact unless the project accepts generated-buffer regeneration as parameter-equivalent.",
        "- `NONEXACT_MISSING_TRAINABLE_PARAMETER` is not parameter-exact because a trainable parameter is initialized rather than loaded.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python-bin", default=PYTHON_BIN)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602")
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()

    probes = [
        (
            "redet_result_config_original_checkpoint",
            args.repo_root / "M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py",
            args.repo_root / "weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth",
        ),
        (
            "redet_result_config_converted_checkpoint",
            args.repo_root / "M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py",
            args.repo_root / "weights/integrity_converted/ReDet_re50_refpn_1x_dota1-a025e6b1_currentkeys.pth",
        ),
        (
            "redet_result_config_converted_materialized_checkpoint",
            args.repo_root / "M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py",
            args.repo_root / "weights/integrity_converted/ReDet_re50_refpn_1x_dota1-a025e6b1_currentkeys_materialized_e2cnn_buffers.pth",
        ),
        (
            "redet_official_repro_original_checkpoint",
            args.repo_root / "M_configs/OfficialMMRotateWeightRepro/redet/redet-le90_re50_refpn_1x_dota.py",
            args.repo_root / "weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth",
        ),
        (
            "redet_msrr_registered_checkpoint",
            args.repo_root / "mmrotate_configs/redet/redet-le90_re50_refpn_rr-1x_dota-ms.py",
            args.repo_root / "weights/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth",
        ),
        (
            "redet_msrr_registered_materialized_checkpoint",
            args.repo_root / "mmrotate_configs/redet/redet-le90_re50_refpn_rr-1x_dota-ms.py",
            args.repo_root / "weights/integrity_converted/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5_materialized_e2cnn_buffers.pth",
        ),
        (
            "oriented_reppoints_registered_checkpoint",
            args.repo_root / "M_configs/OfficialMMRotateWeightRepro/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py",
            args.repo_root / "weights/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth",
        ),
    ]
    rows = []
    for name, config, checkpoint in probes:
        rows.append(run_probe(args, name, config, checkpoint))
    write_csv(args.output_dir / "remaining_model_exact_load_probe.csv", rows)
    (args.output_dir / "remaining_model_exact_load_probe.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_md(args.output_dir / "remaining_model_exact_load_probe.md", rows)
    print(f"wrote={args.output_dir / 'remaining_model_exact_load_probe.md'}")


if __name__ == "__main__":
    main()

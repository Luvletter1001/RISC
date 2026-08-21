#!/usr/bin/env python
"""Build a readiness/evidence gate for G3-v2/BASS matched-control experiments.

This script does not launch training.  It checks whether the matched-control
experiment entry point exists and whether any current artifact proves a
positive G3-v2/BASS result.  Missing results remain a failed method gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_G3_AUDIT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "g3_v1_dense_head_audit.json")
DEFAULT_G3_CFG = Path(
    "M_configs/Diagnostics/"
    "hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_v2_apsensitive_train.py")
DEFAULT_CTRL_CFG = Path(
    "M_configs/Diagnostics/"
    "hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_v2_nog3_control_train.py")
DEFAULT_TEST = Path("tests/test_g3_v2_config_gate.py")
DEFAULT_OUT_DIR = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "g3v2_bass_readiness")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_g3v2_bass_readiness_gate.md")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def parse_config_text(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    def find_int(pattern):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    def find_float(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    work_dir_match = re.search(r"work_dir\s*=\s*\((.*?)\)", text, re.S)
    work_dir = ""
    if work_dir_match:
        parts = re.findall(r"['\"]([^'\"]+)['\"]", work_dir_match.group(1))
        work_dir = "".join(parts)

    return {
        "path": str(path),
        "exists": bool(text),
        "work_dir": work_dir,
        "seed": find_int(r"seed\s*=\s*([0-9]+)"),
        "max_epochs": find_int(r"max_epochs\s*=\s*([0-9]+)"),
        "loss_weight": find_float(r"loss_weight\s*=\s*([0-9.]+)"),
        "min_hardneg_logit": find_float(r"min_hardneg_logit\s*=\s*([-0-9.]+)"),
        "max_gt_abs_z": find_float(r"max_gt_abs_z\s*=\s*([0-9.]+)"),
        "enable_false": "enable=False" in text or "enable = False" in text,
        "consistency_enable_false": (
            "consistency_loss=dict(\n                enable=False" in text
            or "consistency_loss=dict(enable=False" in text),
    }


def path_has_artifacts(work_dir):
    if not work_dir:
        return {
            "work_dir_exists": False,
            "checkpoint_exists": False,
            "prediction_exists": False,
            "eval_json_exists": False,
        }
    path = Path(work_dir)
    return {
        "work_dir_exists": path.exists(),
        "checkpoint_exists": any(path.glob("epoch_*.pth")) if path.exists() else False,
        "prediction_exists": any(path.glob("**/predictions.pkl")) if path.exists() else False,
        "eval_json_exists": any(path.glob("**/*.json")) if path.exists() else False,
    }


def build_markdown(path, payload):
    g3 = payload["g3_config"]
    ctrl = payload["control_config"]
    audit = payload["g3_v1_audit_summary"]
    lines = [
        "# G3-v2 / BASS Readiness Gate - 2026-06-20",
        "",
        "## 结论",
        "",
        f"- `matched_config_ready`: `{str(payload['matched_config_ready']).lower()}`",
        f"- `g3v2_experiment_started`: `{str(payload['g3v2_experiment_started']).lower()}`",
        f"- `g3v2_positive_gate_pass`: `{str(payload['g3v2_positive_gate_pass']).lower()}`",
        f"- `method_gate_blocker`: `{payload['method_gate_blocker']}`",
        "",
        "这个 gate 只检查 G3-v2/BASS 是否具备正确 matched-control 实验入口，",
        "不启动 GPU，也不把未跑完/未阳性的实验写成结果。",
        "",
        "## Matched Config Check",
        "",
        "| item | G3-v2 | no-G3 control |",
        "|---|---:|---:|",
        f"| `exists` | `{g3['exists']}` | `{ctrl['exists']}` |",
        f"| `seed` | `{g3['seed']}` | `{ctrl['seed']}` |",
        f"| `max_epochs` | `{g3['max_epochs']}` | `{ctrl['max_epochs']}` |",
        f"| `work_dir` | `{g3['work_dir']}` | `{ctrl['work_dir']}` |",
        f"| `loss_weight` | `{g3['loss_weight']}` | `{ctrl['loss_weight']}` |",
        f"| `min_hardneg_logit` | `{g3['min_hardneg_logit']}` | `` |",
        f"| `max_gt_abs_z` | `{g3['max_gt_abs_z']}` | `` |",
        "",
        "## Current Evidence",
        "",
        "| source | value |",
        "|---|---|",
        f"| G3-v1 status | `{audit.get('status')}` |",
        f"| best G3 beats best no-G3 mAP | `{audit.get('best_g3_beats_best_nog3_map')}` |",
        f"| best no-G3 mAP | `{audit.get('best_nog3_mAP')}` |",
        f"| best G3 mAP | `{audit.get('best_g3_mAP')}` |",
        "",
        "## Next Gate",
        "",
        "必须运行 G3-v2 与 no-G3 control 的 same-seed/same-epoch 对照，并用",
        "`mAP`, `AP50`, classwise AP, SISE, rank-tail, score-only calibration 对照",
        "共同判定。若不能超过 matched no-G3，只能继续写成 negative/partial。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g3-audit-json", default=str(DEFAULT_G3_AUDIT_JSON))
    parser.add_argument("--g3-config", default=str(DEFAULT_G3_CFG))
    parser.add_argument("--control-config", default=str(DEFAULT_CTRL_CFG))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    args = parser.parse_args()

    g3_audit = read_json(Path(args.g3_audit_json)) or {}
    g3_config = parse_config_text(Path(args.g3_config))
    ctrl_config = parse_config_text(Path(args.control_config))
    g3_artifacts = path_has_artifacts(g3_config["work_dir"])
    ctrl_artifacts = path_has_artifacts(ctrl_config["work_dir"])

    matched_config_ready = (
        g3_config["exists"]
        and ctrl_config["exists"]
        and Path(args.test_file).exists()
        and g3_config["seed"] == ctrl_config["seed"]
        and g3_config["max_epochs"] == ctrl_config["max_epochs"]
        and g3_config["loss_weight"] and g3_config["loss_weight"] > 0
        and ctrl_config["enable_false"]
    )
    g3v2_experiment_started = (
        g3_artifacts["work_dir_exists"] or ctrl_artifacts["work_dir_exists"])
    best_nog3 = g3_audit.get("best_nog3", {}) or {}
    best_g3 = g3_audit.get("best_g3_by_map", {}) or {}
    g3v2_positive_gate_pass = False
    method_gate_blocker = (
        "matched configs ready, but no positive G3-v2/BASS matched-control "
        "result is available")
    payload = {
        "matched_config_ready": bool(matched_config_ready),
        "g3v2_experiment_started": bool(g3v2_experiment_started),
        "g3v2_positive_gate_pass": g3v2_positive_gate_pass,
        "method_gate_blocker": method_gate_blocker,
        "g3_config": g3_config,
        "control_config": ctrl_config,
        "g3_artifacts": g3_artifacts,
        "control_artifacts": ctrl_artifacts,
        "test_file": args.test_file,
        "g3_v1_audit_summary": {
            "status": g3_audit.get("status"),
            "best_g3_beats_best_nog3_map": g3_audit.get(
                "best_g3_beats_best_nog3_map"),
            "best_nog3_mAP": best_nog3.get("mAP"),
            "best_g3_mAP": best_g3.get("mAP"),
            "result_md": g3_audit.get("result_md"),
        },
        "required_next_evidence": [
            "G3-v2 eval JSON",
            "matched no-G3 eval JSON",
            "deployment risk summary for both variants",
            "rank-tail paired audit",
            "classwise AP or AP-sensitive utility table",
        ],
        "result_md": str(Path(args.result_md)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "g3v2_bass_readiness_gate.json", payload)
    build_markdown(Path(args.result_md), payload)
    print(json.dumps({
        "matched_config_ready": payload["matched_config_ready"],
        "g3v2_experiment_started": payload["g3v2_experiment_started"],
        "g3v2_positive_gate_pass": payload["g3v2_positive_gate_pass"],
        "json": str(out_dir / "g3v2_bass_readiness_gate.json"),
        "md": str(Path(args.result_md)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

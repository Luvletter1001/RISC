#!/usr/bin/env python3
"""Wrapper for large-scale live OpenRSD hook diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from strict_gpu_runner_v3 import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--angles", default="000,030,060,090,120,150,180,210,240,270,300,330")
    parser.add_argument("--heads", default="alignment,fusion")
    parser.add_argument("--prompt-family", default="F3_orientation_aware")
    parser.add_argument("--max-live-images", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data1/zcy/OpenRSD/results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth"))
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/data/zcy/anaconda3/envs/openrsd/bin/python"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hook_dir = args.work_dir / "G4_hooks"
    csv_path = hook_dir / "live_hook_diagnostic.csv"
    json_path = hook_dir / "live_hook_summary.json"
    cmd = [
        str(args.python_bin if args.python_bin.exists() else Path(sys.executable)),
        "M_Tools/analysis/run_ovd_feature_text_logit_hooks.py",
        "--repo-root", str(args.repo_root),
        "--existing-work-dir", str(args.repo_root / "work_dirs/openrsd_ovd_rotation_20260508"),
        "--prompt-family", args.prompt_family,
        "--angles", args.angles,
        "--heads", args.heads,
        "--max-live-images", str(args.max_live_images),
        "--checkpoint", str(args.checkpoint),
        "--live-work-dir", str(hook_dir / "live_work"),
        "--live-hooks",
        "--out-csv", str(csv_path),
        "--out-json", str(json_path),
    ]
    env = {
        **dict(**__import__("os").environ),
        "PYTHONNOUSERSITE": "1",
        "MPLCONFIGDIR": "/tmp/mplconfig",
        "CUDA_VISIBLE_DEVICES": args.gpu_ids,
        "PYTHONPATH": f"{args.repo_root}:{args.repo_root / 'tools'}",
    }
    hook_dir.mkdir(parents=True, exist_ok=True)
    stdout = hook_dir / "wrapper_stdout.log"
    stderr = hook_dir / "wrapper_stderr.log"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(["rtk", "env", *[f"{k}={v}" for k, v in env.items() if k in {"PYTHONNOUSERSITE", "MPLCONFIGDIR", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"}], *cmd], cwd=str(args.repo_root), stdout=out, stderr=err, text=True, check=False)
    summary = {}
    if json_path.exists():
        summary = json.loads(json_path.read_text(encoding="utf-8"))
    status = "DONE_LIVE_GPU" if proc.returncode == 0 and csv_path.exists() and summary.get("row_count", 0) >= 1 else "FAILED"
    payload = {
        "status": status,
        "return_code": proc.returncode,
        "csv": str(csv_path),
        "json": str(json_path),
        "row_count": summary.get("row_count"),
        "successful_hooks": summary.get("successful_hooks", []),
        "failed_or_missing_hooks": summary.get("failed_or_missing_hooks", []),
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    write_json(args.out_json, payload)
    lines = [
        "# G4 Large-Scale Live Hook Diagnostic",
        "",
        f"- status: `{status}`",
        f"- RUN_TS: `{args.run_ts}`",
        f"- angles: `{args.angles}`",
        f"- heads: `{args.heads}`",
        f"- max_live_images_per_angle: `{args.max_live_images}`",
        f"- csv: `{csv_path}`",
        f"- json: `{json_path}`",
        f"- row_count: `{summary.get('row_count')}`",
        f"- successful_hooks: `{summary.get('successful_hooks', [])}`",
        f"- failed_hooks: `{summary.get('failed_or_missing_hooks', [])}`",
        "",
        "This wrapper invokes live forward hooks; it does not reuse old hook CSV files.",
    ]
    write_text(args.out_md, "\n".join(lines))


if __name__ == "__main__":
    main()


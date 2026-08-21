#!/usr/bin/env python3
"""Probe DOTA1 train split and optionally launch a short training smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gpu45_task_runner import append_jsonl, result_to_dict, run_child, shell_join, write_json, write_text  # noqa: E402


TRAIN_CANDIDATES = [
    "data/DOTA1_1024_500/ss_train",
    "data/DOTA1_1024_500/train",
    "data/DOTA1_1024_500/trainval",
    "data/DOTA1_1024_500/trainval_split",
    "data/DOTA1_1024_500/trainval1024",
    "data/DOTA1_1024_500/split_ss_train",
]


def has_train_split(path: Path) -> bool:
    return (
        (path / "images").is_dir()
        and ((path / "Step6_Format_labels").is_dir() or (path / "annfiles").is_dir() or (path / "labelTxt").is_dir())
    )


def discover(repo_root: Path) -> dict[str, Any]:
    candidates = []
    for rel in TRAIN_CANDIDATES:
        path = repo_root / rel
        candidates.append({"path": str(path), "exists": path.exists(), "usable": has_train_split(path)})
    usable = next((Path(item["path"]) for item in candidates if item["usable"]), None)
    raw_zip = repo_root / "data/DOTAV1/train/labelTxt.zip"
    raw_external = Path("/data/zcy/dataset/dota15")
    split_tools = [
        repo_root / "M_Tools/analysis/prepare_dota1_1024_500_split.py",
        repo_root / "DOTA_devkit/ImgSplit_multi_process.py",
    ]
    return {
        "candidates": candidates,
        "usable_train_root": str(usable) if usable else "",
        "raw_label_zip": str(raw_zip),
        "raw_label_zip_exists": raw_zip.exists(),
        "raw_external": str(raw_external),
        "raw_external_exists": raw_external.exists(),
        "split_tools": [{"path": str(p), "exists": p.exists()} for p in split_tools],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    info = discover(args.repo_root)
    child_results: list[dict[str, Any]] = []
    planned = []
    if info["usable_train_root"]:
        planned.append([
            str(args.python_bin), "M_Tools/analysis/write_openrsd_dota1_fix_configs.py",
            "--repo-root", str(args.repo_root),
            "--train-root", info["usable_train_root"],
            "--work-dir", str(args.work_dir / "P5_training_probe"),
            "--max-iters", str(args.max_iters),
            "--batch-size", str(args.batch_size),
        ])
    if args.mode == "dryrun":
        payload = {"status": "DRYRUN", "discovery": info, "planned_commands": [shell_join(cmd) for cmd in planned]}
        write_json(args.out_json, payload)
        write_text(args.out_md, "# P5 Train Split Repair And Training Probe\n\n- status: `DRYRUN`\n")
        return payload
    if not info["usable_train_root"]:
        status = "PARTIAL"
        payload = {
            "status": status,
            "discovery": info,
            "reason": "No usable DOTA1_1024_500 train split found; training probe not launched.",
            "child_results": child_results,
        }
    else:
        cfg_result = run_child(
            task_name="P5_write_train_configs",
            argv=planned[0],
            repo_root=args.repo_root,
            log_dir=args.work_dir / "P5_logs/P5_write_train_configs",
            gpu_ids=args.gpu_ids,
        )
        cfg_payload = result_to_dict(cfg_result)
        child_results.append(cfg_payload)
        append_jsonl(args.work_dir / "commands.jsonl", cfg_payload)
        manifest_path = args.work_dir / "P5_training_probe/config_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if args.mode in {"smoke", "full", "debug"} and cfg_result.return_code == 0 and manifest.get("f3_config"):
            train_cmd = [
                str(args.python_bin), "tools/train.py",
                manifest["f3_config"],
                "--work-dir", str(args.work_dir / "P5_training_probe/F3_baseline_short_ft"),
            ]
            if args.mode != "full":
                train_cmd.extend(["--cfg-options", f"max_iter_per_epoch={args.max_iters}"])
            train_result = run_child(
                task_name="P5_baseline_short_ft_train",
                argv=train_cmd,
                repo_root=args.repo_root,
                log_dir=args.work_dir / "P5_logs/P5_baseline_short_ft_train",
                gpu_ids=args.gpu_ids,
                timeout_sec=3600 if args.mode != "full" else None,
            )
            train_payload = result_to_dict(train_result)
            child_results.append(train_payload)
            append_jsonl(args.work_dir / "commands.jsonl", train_payload)
            if train_result.return_code:
                append_jsonl(args.work_dir / "failures.jsonl", train_payload)
        status = "DONE" if child_results and all(r.get("return_code") == 0 for r in child_results) else "PARTIAL"
        payload = {"status": status, "discovery": info, "manifest": manifest, "child_results": child_results}
    write_json(args.out_json, payload)
    lines = [
        "# P5 Train Split Repair And Training Probe",
        "",
        f"- generated_at: `{args.run_ts}`",
        f"- status: `{payload['status']}`",
        f"- usable_train_root: `{info.get('usable_train_root') or 'NA'}`",
        f"- raw_label_zip: `{info.get('raw_label_zip')}` exists `{info.get('raw_label_zip_exists')}`",
        f"- raw_external: `{info.get('raw_external')}` exists `{info.get('raw_external_exists')}`",
        "",
        "## Candidate Train Roots",
        "",
        "| path | exists | usable |",
        "|---|---:|---:|",
    ]
    for item in info["candidates"]:
        lines.append(f"| `{item['path']}` | {item['exists']} | {item['usable']} |")
    lines.extend(["", "## Commands", ""])
    for result in child_results:
        lines.append(f"- `{result.get('command')}` rc=`{result.get('return_code')}` stdout=`{result.get('stdout_path')}` stderr=`{result.get('stderr_path')}`")
    if not info["usable_train_root"]:
        lines.extend(["", "## Conclusion", "", "- PARTIAL: no training was launched, so this cannot be used as method effectiveness evidence."])
    else:
        lines.extend(["", "## Conclusion", "", "- A usable train split was found and a training probe was attempted. Check child logs for loss/checkpoint details."])
    write_text(args.out_md, "\n".join(lines))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    default_python = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--python-bin", type=Path, default=default_python if default_python.exists() else Path(sys.executable))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

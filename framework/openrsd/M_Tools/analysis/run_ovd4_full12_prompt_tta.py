#!/usr/bin/env python3
"""GPU45 wrapper for OVD4 full-12 prompt ensemble + rotation TTA."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gpu45_task_runner import append_jsonl, result_to_dict, run_child, shell_join, write_json, write_text  # noqa: E402


ANGLES12 = "000,030,060,090,120,150,180,210,240,270,300,330"
ANGLES_SMOKE = "000,090"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = []
    for row in rows:
        try:
            vals.append((row.get("target_angle") or row.get("angle"), float(row["ap50"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return {}
    aps = [v for _, v in vals]
    worst = min(vals, key=lambda x: x[1])
    best = max(vals, key=lambda x: x[1])
    mean = statistics.mean(aps)
    return {
        "mean": mean,
        "worst": worst[1],
        "worst_angle": worst[0],
        "best": best[1],
        "best_angle": best[0],
        "std": statistics.pstdev(aps) if len(aps) > 1 else 0.0,
        "range": best[1] - worst[1],
        "rsi": worst[1] / mean if mean else None,
    }


def build_suite_cmd(args: argparse.Namespace, exp: str, mode: str, angles: str, max_images: int | None, gpu_ids: str) -> list[str]:
    cmd = [
        str(args.python_bin),
        "M_Tools/analysis/run_openrsd_ovd_rotation_suite.py",
        "--repo-root", str(args.repo_root),
        "--result-md-dir", str(args.result_md_dir),
        "--weights-dir", str(args.weights_dir),
        "--work-dir", str(args.work_dir / "P1_ovd_suite"),
        "--gpu-ids", gpu_ids,
        "--exp", exp,
        "--mode", mode,
        "--only-angle", angles,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--resume",
    ]
    if max_images is not None:
        cmd.extend(["--max-images-full", str(max_images), "--max-images-for-smoke", str(max_images)])
    return cmd


def summarize(args: argparse.Namespace, child_results: list[dict[str, Any]], dryrun_commands: list[str]) -> dict[str, Any]:
    report_path = args.work_dir / "P1_ovd_suite/exp_ovd4/exp_ovd4_results.json"
    report = load_json(report_path)
    source_note = "current work_dir"
    if not report.get("rows"):
        fallback_path = args.existing_work_dir / "exp_ovd4/exp_ovd4_results.json"
        fallback = load_json(fallback_path)
        if fallback.get("rows"):
            report = fallback
            source_note = str(fallback_path)
    rows = report.get("rows", [])
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(row.get("strategy", "unknown"), []).append(row)
    summary = {key: stats(val) for key, val in by_strategy.items()}
    status = report.get("status") or ("DONE" if rows else ("DRYRUN" if args.mode == "dryrun" else "FAILED"))
    if rows and status == "NOT_RUN":
        status = "DONE" if args.mode in {"smoke", "debug"} else "PARTIAL"
    if any(item.get("return_code") not in (0, None) for item in child_results):
        status = "PARTIAL" if rows else "FAILED"
    payload = {
        "status": status,
        "level": "Level 1 target-12/source-from-available-OVD2" if args.mode == "full" else "smoke target/source 000,090",
        "source_angles": report.get("source_angles", []),
        "target_angles": report.get("target_angles", []),
        "best_prompt_family": report.get("best_prompt_family"),
        "summary": summary,
        "rows": rows,
        "ablation_rows": report.get("ablation_rows", []),
        "child_results": child_results,
        "dryrun_commands": dryrun_commands,
        "report_json": str(report_path),
        "source_note": source_note,
    }
    write_json(args.out_json, payload)
    lines = [
        "# P1 OVD4 Full-12 Prompt Ensemble + Rotation TTA",
        "",
        f"- generated_at: `{args.run_ts}`",
        f"- status: `{status}`",
        f"- work_dir: `{args.work_dir}`",
        f"- ovd_suite_work_dir: `{args.work_dir / 'P1_ovd_suite'}`",
        f"- level: `{payload['level']}`",
        f"- source_angles: `{payload['source_angles']}`",
        f"- target_angles: `{payload['target_angles']}`",
        f"- prompt families: `F0_raw_class,F1_aerial_context,F2_remote_sensing_context,F3_orientation_aware,F4_shape_aware`",
        f"- best_prompt_family: `{payload.get('best_prompt_family')}`",
        f"- source_note: `{source_note}`",
        f"- head: `alignment`; fusion is delegated to OVD3/P2 if available",
        "",
        "## Strategy Summary",
        "",
        "| strategy | mean AP50 | worst AP50 | worst angle | best AP50 | best angle | std | range | RSI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, item in sorted(summary.items()):
        lines.append(
            f"| {strategy} | {fmt(item.get('mean'))} | {fmt(item.get('worst'))} | {item.get('worst_angle', 'NA')} | "
            f"{fmt(item.get('best'))} | {item.get('best_angle', 'NA')} | {fmt(item.get('std'))} | "
            f"{fmt(item.get('range'))} | {fmt(item.get('rsi'))} |"
        )
    lines.extend([
        "",
        "## Per-Angle AP50",
        "",
        "| strategy | target_angle | status | AP50 | det/img | predictions | stdout | stderr |",
        "|---|---:|---|---:|---:|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row.get('strategy')} | {row.get('target_angle')} | {row.get('status')} | {fmt(row.get('ap50'))} | "
            f"{fmt(row.get('avg_detections_per_image'))} | `{row.get('predictions', '')}` | "
            f"`{row.get('stdout', '')}` | `{row.get('stderr', '')}` |"
        )
    lines.extend([
        "",
        "## NMS / Score Ablation",
        "",
        "| target_angle | score_thr | nms_iou | score_rule | status | AP50 | det/img | merged_pkl |",
        "|---:|---:|---:|---|---|---:|---:|---|",
    ])
    for row in report.get("ablation_rows", []):
        lines.append(
            f"| {row.get('target_angle')} | {row.get('score_thr')} | {row.get('nms_iou')} | {row.get('score_rule')} | "
            f"{row.get('status')} | {fmt(row.get('ap50'))} | {fmt(row.get('avg_detections_per_image'))} | `{row.get('merged_pkl', '')}` |"
        )
    lines.extend([
        "",
        "## Commands",
        "",
    ])
    for cmd in dryrun_commands:
        lines.append(f"- `{cmd}`")
    for result in child_results:
        lines.append(f"- `{result.get('command')}`")
    failed = [r for r in child_results if r.get("return_code")]
    lines.extend(["", "## Failed Commands", ""])
    if failed:
        for result in failed:
            lines.append(f"- `{result.get('task_name')}` rc={result.get('return_code')} stderr=`{result.get('stderr_path')}` failure=`{result.get('failure_kind')}`")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The wrapper uses the existing OpenRSD OVD suite and does not relabel 4-view TTA as 12-view TTA.",
        "- Full target-12 coverage depends on OVD2 rows for all requested source angles; missing source rows are reflected in the table and logs.",
    ])
    write_text(args.out_md, "\n".join(lines))
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "P1_logs").mkdir(parents=True, exist_ok=True)
    dryrun_commands: list[str] = []
    child_results: list[dict[str, Any]] = []
    angles = ANGLES_SMOKE if args.mode == "smoke" else ANGLES12
    max_images = args.max_images_for_smoke if args.mode == "smoke" else args.max_images_full
    planned = [
        [str(args.python_bin), "M_Tools/analysis/run_openrsd_ovd_rotation_suite.py", "--exp", "ovd1", "--mode", "smoke"],
        build_suite_cmd(args, "ovd2", "debug" if args.mode == "smoke" else "full", angles, max_images, args.gpu_ids),
        build_suite_cmd(args, "ovd4", "debug" if args.mode == "smoke" else "full", angles, max_images, args.gpu_ids),
    ]
    dryrun_commands = [shell_join(cmd) for cmd in planned]
    if args.mode == "dryrun":
        return summarize(args, child_results, dryrun_commands)

    smoke_cmd = [
        str(args.python_bin), "M_Tools/analysis/run_openrsd_ovd_rotation_suite.py",
        "--repo-root", str(args.repo_root),
        "--result-md-dir", str(args.result_md_dir),
        "--weights-dir", str(args.weights_dir),
        "--work-dir", str(args.work_dir / "P1_ovd_suite"),
        "--gpu-ids", "4",
        "--exp", "ovd1",
        "--mode", "smoke",
        "--batch-size", "1",
        "--max-images-for-smoke", str(args.max_images_for_smoke),
        "--resume",
    ]
    for task_name, cmd, gpu_ids in [
        ("P1_preflight_ovd_smoke", smoke_cmd, "4"),
        ("P1_ovd2_prompt_family", planned[1], args.gpu_ids),
        ("P1_ovd4_prompt_tta", planned[2], args.gpu_ids),
    ]:
        result = run_child(task_name=task_name, argv=cmd, repo_root=args.repo_root, log_dir=args.work_dir / "P1_logs" / task_name, gpu_ids=gpu_ids)
        payload = result_to_dict(result)
        child_results.append(payload)
        append_jsonl(args.work_dir / "commands.jsonl", payload)
        if result.return_code:
            append_jsonl(args.work_dir / "failures.jsonl", payload)
            if not args.fallback_on_failure:
                break
    return summarize(args, child_results, dryrun_commands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--existing-work-dir", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508"))
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images-for-smoke", type=int, default=16)
    parser.add_argument("--max-images-full", type=int, default=None)
    parser.add_argument("--fallback-on-failure", action="store_true")
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

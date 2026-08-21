#!/usr/bin/env python3
"""Run live OpenRSD inference on DOTA1 angle sweep without reusing old pkl."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_openrsd_ovd_rotation_suite as suite  # noqa: E402
from strict_gpu_runner_v3 import write_json, write_text  # noqa: E402


ANGLES12 = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]


def parse_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(v):
        return "NA"
    return f"{v:.4f}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = []
    for r in rows:
        if r.get("status") != "OK":
            continue
        metric = r.get("ap50")
        if metric is None:
            metric = r.get("map")
        try:
            vals.append((r.get("angle"), float(metric)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {}
    aps = [v for _, v in vals]
    worst = min(vals, key=lambda x: x[1])
    best = max(vals, key=lambda x: x[1])
    mean = statistics.mean(aps)
    return {
        "n": len(vals),
        "mean": mean,
        "worst": worst[1],
        "worst_angle": worst[0],
        "best": best[1],
        "best_angle": best[0],
        "std": statistics.pstdev(aps) if len(aps) > 1 else 0.0,
        "range": best[1] - worst[1],
        "rsi": worst[1] / mean if mean else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--angles", default=",".join(ANGLES12))
    parser.add_argument("--prompt-families", default="F3_orientation_aware")
    parser.add_argument("--heads", default="alignment")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/data/zcy/anaconda3/envs/openrsd/bin/python"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    angles = [f"{int(a):03d}" for a in parse_list(args.angles, ANGLES12)]
    prompt_families = parse_list(args.prompt_families, ["F3_orientation_aware"])
    heads = parse_list(args.heads, ["alignment"])
    config = args.repo_root / "M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py"
    ckpt = args.weights_dir / "MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth"
    model = suite.ModelChoice("openrsd_a12", config, ckpt, 100, "fixed requested OpenRSD checkpoint")
    suite_args = argparse.Namespace(
        repo_root=args.repo_root,
        result_md_dir=args.result_md_dir,
        weights_dir=args.weights_dir,
        work_dir=args.work_dir,
        gpu_ids=args.gpu_ids,
        force=True,
        resume=False,
        python_bin=args.python_bin if args.python_bin.exists() else Path(sys.executable),
    )
    runner = suite.CommandRunner(suite_args)
    rows: list[dict[str, Any]] = []
    desired_records = []
    max_images = args.max_images if args.max_images and args.max_images > 0 else None
    for prompt in prompt_families:
        prompt_texts = suite.PROMPT_FAMILIES_OVD2.get(prompt)
        if not prompt_texts:
            rows.append({"status": "FAILED", "prompt_key": prompt, "error": "unknown prompt family"})
            continue
        class_names = suite.CORE9
        for head in heads:
            for angle in angles:
                row = suite.run_single_inference(
                    args=suite_args,
                    runner=runner,
                    model=model,
                    angle=angle,
                    prompt_key=prompt,
                    prompt_texts=prompt_texts,
                    class_names=class_names,
                    head=head,
                    exp_dir=args.work_dir / "G2_live_inference_generated",
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_images=max_images,
                    gpu_ids=args.gpu_ids,
                    distributed=False,
                )
                src_pred = Path(row.get("predictions", ""))
                desired_dir = args.work_dir / "G2_live_inference" / head / prompt / f"angle_{angle}"
                desired_dir.mkdir(parents=True, exist_ok=True)
                desired_pred = desired_dir / "predictions.pkl"
                if src_pred.exists():
                    shutil.copy2(src_pred, desired_pred)
                    row["formal_predictions"] = str(desired_pred)
                    row["formal_predictions_mtime"] = desired_pred.stat().st_mtime
                row["live_file_newer_than_run_ts"] = bool(src_pred.exists() and row.get("formal_predictions_mtime"))
                rows.append(row)
                desired_records.append(str(desired_pred))
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(f"{row.get('head')}|{row.get('prompt_key')}", []).append(row)
    group_summary = {k: summarize(v) for k, v in by_group.items()}
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    live_gpu_rows = [
        r for r in ok_rows
        if max((r.get("peak_mem_mb") or {"0": 0}).values() or [0]) >= 2000 and any((r.get("process_seen") or {}).values())
    ]
    status = "DONE_LIVE_GPU" if len(ok_rows) == len(rows) and len(live_gpu_rows) == len(ok_rows) else ("PARTIAL_LIVE_GPU" if live_gpu_rows else "GPU_NOT_USED")
    payload = {
        "status": status,
        "run_ts": args.run_ts,
        "angles": angles,
        "prompt_families": prompt_families,
        "heads": heads,
        "batch_size": args.batch_size,
        "max_images": max_images,
        "rows": rows,
        "group_summary": group_summary,
        "formal_prediction_paths": desired_records,
    }
    write_json(args.out_json, payload)
    if args.out_md:
        lines = [
            "# G2 Live Full-12 OpenRSD Inference",
            "",
            f"- status: `{status}`",
            f"- RUN_TS: `{args.run_ts}`",
            f"- work_dir: `{args.work_dir}`",
            f"- angles: `{angles}`",
            f"- prompt_families: `{prompt_families}`",
            f"- heads: `{heads}`",
            f"- batch_size: `{args.batch_size}`",
            f"- max_images: `{max_images}`",
            "- cache policy: `force_live_no_old_pkl_as_result`",
            "",
            "## Group Summary",
            "",
            "| group | n | mean | worst | worst_angle | best | best_angle | std | range | RSI |",
            "|---|---:|---:|---:|---|---:|---|---:|---:|---:|",
        ]
        for group, item in sorted(group_summary.items()):
            lines.append(
                f"| {group} | {item.get('n', 0)} | {fmt(item.get('mean'))} | {fmt(item.get('worst'))} | {item.get('worst_angle', 'NA')} | "
                f"{fmt(item.get('best'))} | {item.get('best_angle', 'NA')} | {fmt(item.get('std'))} | {fmt(item.get('range'))} | {fmt(item.get('rsi'))} |"
            )
        lines.extend(["", "## Per Angle", "", "| head | prompt | angle | status | AP50 | mAP | samples | peak_mem | process_seen | predictions | stdout | stderr |", "|---|---|---:|---|---:|---:|---:|---|---|---|---|---|"])
        for r in rows:
            peak = r.get("peak_mem_mb")
            lines.append(
                f"| {r.get('head')} | {r.get('prompt_key')} | {r.get('angle')} | {r.get('status')} | {fmt(r.get('ap50'))} | {fmt(r.get('map'))} | "
                f"{r.get('sample_count', 'NA')} | `{peak}` | `{r.get('process_seen')}` | `{r.get('formal_predictions') or r.get('predictions')}` | `{r.get('stdout')}` | `{r.get('stderr')}` |"
            )
        write_text(args.out_md, "\n".join(lines))


if __name__ == "__main__":
    main()


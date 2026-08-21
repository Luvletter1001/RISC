#!/usr/bin/env python3
"""Clean wrapper for geometry per-bin AP with separated closed/OVD settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_per_bin_ap import run_eval  # noqa: E402
from gpu45_task_runner import write_json, write_text  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_prediction_rows(existing_work_dir: Path, angle: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("ovd_alignment", existing_work_dir / "exp_ovd3/exp_ovd3_results.json"),
        ("ovd_prompt", existing_work_dir / "exp_ovd2/exp_ovd2_results.json"),
        ("ovd_tta", existing_work_dir / "exp_ovd4/exp_ovd4_results.json"),
        ("ovd_zero_shot", existing_work_dir / "exp_ovd1/exp_ovd1_results.json"),
    ]
    for setting, path in sources:
        report = load_json(path)
        for row in report.get("rows", []):
            row_angle = row.get("angle") or row.get("target_angle")
            pred = row.get("predictions")
            if row.get("status") == "OK" and row_angle == angle and pred and Path(pred).exists():
                rows.append({"setting": setting, "source_json": str(path), **row})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "definition", "setting", "model", "angle", "bin_type", "bin", "ap50",
        "gt_count", "prediction_count", "tp", "fp", "fn", "mean_score", "predictions"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_one(args: argparse.Namespace, source: dict[str, Any], idx: int) -> dict[str, Any]:
    out_json = args.work_dir / "P4_geometry" / f"{idx:03d}_{source['setting']}_angle_{args.angle}.json"
    out_csv = args.work_dir / "P4_geometry" / f"{idx:03d}_{source['setting']}_angle_{args.angle}.csv"
    eval_args = argparse.Namespace(
        repo_root=args.repo_root,
        predictions=Path(source["predictions"]),
        angle=args.angle,
        model=source.get("setting") or source.get("model") or "unknown",
        classes=",".join(source.get("per_class_ap50", {}).keys()) if source.get("per_class_ap50") else "",
        out_csv=out_csv,
        out_json=out_json,
    )
    payload = run_eval(eval_args)
    for row in payload.get("rows", []):
        row["definition"] = "A_GT_bin_AP"
        row["setting"] = source["setting"]
        row["model"] = eval_args.model
        row["angle"] = args.angle
        row["predictions"] = source["predictions"]
    # Definition B needs prediction-bin filtering. Keep the result explicit when
    # the underlying evaluator has not produced a joint-bin pass yet.
    joint_rows = []
    for row in payload.get("rows", []):
        copy = dict(row)
        copy["definition"] = "B_joint_bin_AP_NOT_AVAILABLE"
        copy["ap50"] = ""
        copy["tp"] = ""
        copy["fp"] = ""
        copy["fn"] = ""
        joint_rows.append(copy)
    return {
        "status": "DONE",
        "setting": source["setting"],
        "source_json": source.get("source_json"),
        "predictions": source["predictions"],
        "payload": payload,
        "rows": payload.get("rows", []) + joint_rows,
        "csv": str(out_csv),
        "json": str(out_json),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = find_prediction_rows(args.existing_work_dir, args.angle)
    if args.mode == "smoke":
        rows = rows[:2]
    if args.mode == "dryrun":
        payload = {"status": "DRYRUN", "candidate_prediction_rows": rows[:20]}
        write_json(args.out_json, payload)
        write_text(args.out_md, "# P4 Geometry Per-Bin AP Rerun\n\n- status: `DRYRUN`\n")
        return payload
    reports = []
    all_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            report = run_one(args, row, idx)
            reports.append(report)
            all_rows.extend(report["rows"])
        except Exception as exc:  # noqa: BLE001
            reports.append({"status": "FAILED", "setting": row.get("setting"), "predictions": row.get("predictions"), "error": repr(exc)})
    status = "DONE" if reports and all(r.get("status") == "DONE" for r in reports) else ("PARTIAL" if any(r.get("status") == "DONE" for r in reports) else "FAILED")
    merged_csv = args.work_dir / "P4_geometry/per_bin_ap_clean_merged.csv"
    write_csv(merged_csv, all_rows)
    payload = {"status": status, "angle": args.angle, "reports": reports, "rows": all_rows, "merged_csv": str(merged_csv)}
    write_json(args.out_json, payload)
    lines = [
        "# P4 Clean Geometry Per-Bin AP Rerun",
        "",
        f"- generated_at: `{args.run_ts}`",
        f"- status: `{status}`",
        f"- angle: `{args.angle}`",
        f"- merged_csv: `{merged_csv}`",
        "- Definition A: `GT-bin AP`, implemented by `eval_per_bin_ap.py`.",
        "- Definition B: `Prediction-GT joint-bin AP`, explicitly marked `NOT_AVAILABLE` until a separate prediction-bin evaluator is implemented.",
        "",
        "| definition | setting | bin_type | bin | AP50 | GT | pred | TP | FP | FN |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row.get('definition')} | {row.get('setting')} | {row.get('bin_type')} | {row.get('bin')} | "
            f"{row.get('ap50', '')} | {row.get('gt_count', '')} | {row.get('prediction_count', '')} | "
            f"{row.get('tp', '')} | {row.get('fp', '')} | {row.get('fn', '')} |"
        )
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        "- Closed-set and OVD rows are not mixed into one model average.",
        "- Definition B remains a reported gap here; do not use this output to claim joint-bin AP improvements.",
    ])
    write_text(args.out_md, "\n".join(lines))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--existing-work-dir", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508"))
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--angle", default="000")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.work_dir = args.work_dir.resolve()
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

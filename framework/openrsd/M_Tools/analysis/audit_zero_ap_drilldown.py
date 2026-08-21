#!/usr/bin/env python3
"""Drill down zero-AP OVD classes with counts, score histograms, and top predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from audit_ovd_zero_ap import audit as base_audit  # noqa: E402
from gpu45_task_runner import write_json, write_text  # noqa: E402


CORE = ["large-vehicle", "plane", "harbor", "helicopter"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def hist(scores: list[float], bins: int = 10) -> list[dict[str, Any]]:
    if not scores:
        return []
    counts, edges = np.histogram(np.asarray(scores, dtype=np.float64), bins=bins, range=(0.0, 1.0))
    return [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]


def extract_predictions(pkl_path: Path, classes: list[str], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class = {name: {"scores": [], "top": []} for name in classes}
    if not pkl_path.exists():
        return {"status": "FAILED", "reason": f"missing predictions {pkl_path}"}
    for sample in load_pickle(pkl_path):
        img_id = str(sample.get("img_id"))
        inst = sample.get("pred_instances", {})
        labels = to_numpy(inst.get("labels", [])).astype(int)
        scores = to_numpy(inst.get("scores", [])).astype(float)
        bboxes = to_numpy(inst.get("bboxes", [])).astype(float)
        for label, score, box in zip(labels, scores, bboxes):
            if 0 <= label < len(classes):
                name = classes[label]
            else:
                continue
            item = per_class.setdefault(name, {"scores": [], "top": []})
            item["scores"].append(float(score))
            item["top"].append({"img_id": img_id, "score": float(score), "bbox": box.tolist()})
    summary = {}
    for name, item in per_class.items():
        top = sorted(item["top"], key=lambda x: x["score"], reverse=True)[:200]
        score_hist = hist(item["scores"])
        (out_dir / f"{name}_top200_predictions.json").write_text(json.dumps(top, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out_dir / f"{name}_score_histogram.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["lo", "hi", "count"])
            writer.writeheader()
            writer.writerows(score_hist)
        summary[name] = {
            "prediction_count": len(item["scores"]),
            "mean_score": float(np.mean(item["scores"])) if item["scores"] else 0.0,
            "top200_json": str(out_dir / f"{name}_top200_predictions.json"),
            "score_histogram_csv": str(out_dir / f"{name}_score_histogram.csv"),
            "iou_histogram_csv": "NOT_AVAILABLE_without_GT_matching",
            "center_distance_histogram_csv": "NOT_AVAILABLE_without_GT_matching",
            "angle_difference_histogram_csv": "NOT_AVAILABLE_without_GT_matching",
        }
    return {"status": "DONE", "classes": summary}


def diagnose(row: dict[str, Any], pred_summary: dict[str, Any]) -> str:
    if row.get("diagnosis") == "NOT_IN_EVALUATOR_CLASS_LIST":
        return "LABEL_MISMATCH"
    if int(row.get("gt_count") or 0) <= 0:
        return "NO_GT"
    if int(row.get("prediction_count") or 0) <= 0:
        return "NO_PREDICTION"
    if float(row.get("mean_confidence") or 0.0) < 0.01:
        return "LOW_SCORE"
    if row.get("ap50") in (0, 0.0, "0", "0.0"):
        return "BAD_IOU_OR_LABEL_MISMATCH"
    return "NONZERO_OR_UNKNOWN"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.work_dir / "P3_zero_ap"
    audit_dir.mkdir(parents=True, exist_ok=True)
    base_args = argparse.Namespace(
        repo_root=args.repo_root,
        existing_work_dir=args.existing_work_dir,
        angle=args.angle,
        prompt_key="F3_orientation_aware",
        out_csv=audit_dir / "zero_ap_audit.csv",
        out_json=audit_dir / "zero_ap_mapping_debug.json",
    )
    if args.mode == "dryrun":
        payload = {
            "status": "DRYRUN",
            "planned_outputs": [str(base_args.out_csv), str(base_args.out_json)],
            "classes": CORE,
        }
        write_json(args.out_json, payload)
        write_text(args.out_md, "# P3 Zero-AP Drilldown\n\n- status: `DRYRUN`\n")
        return payload
    base = base_audit(base_args)
    source_row = base.get("source_row", {})
    class_list = base.get("evaluator_class_list") or list(source_row.get("per_class_ap50", {}).keys())
    pred_path = Path(source_row.get("predictions", ""))
    pred_summary = extract_predictions(pred_path, class_list, audit_dir / "histograms")
    rows = []
    for row in base.get("rows", []):
        if row.get("class") not in CORE:
            continue
        row = dict(row)
        row["drilldown_diagnosis"] = diagnose(row, pred_summary)
        row.update(pred_summary.get("classes", {}).get(row["class"], {}))
        rows.append(row)
    status = "DONE" if base.get("status") == "DONE" else "FAILED"
    payload = {
        "status": status,
        "base_audit": base,
        "prediction_drilldown": pred_summary,
        "rows": rows,
        "csv": str(base_args.out_csv),
        "json": str(base_args.out_json),
    }
    write_json(args.out_json, payload)
    lines = [
        "# P3 Zero-AP Drilldown",
        "",
        f"- generated_at: `{args.run_ts}`",
        f"- status: `{status}`",
        f"- source_predictions: `{pred_path}`",
        f"- audit_csv: `{base_args.out_csv}`",
        f"- mapping_debug_json: `{base_args.out_json}`",
        "",
        "| class | GT | pred | mean conf | AP50 | diagnosis | support norm | text norm | top200 | score hist |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('class')} | {row.get('gt_count')} | {row.get('prediction_count')} | {fmt(row.get('mean_confidence'))} | "
            f"{fmt(row.get('ap50'))} | {row.get('drilldown_diagnosis')} | {fmt(row.get('support_visual_embedding_norm'))} | "
            f"{fmt(row.get('text_embedding_norm'))} | `{row.get('top200_json', '')}` | `{row.get('score_histogram_csv', '')}` |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- IoU / center-distance histograms require a class-specific GT matching pass; when unavailable they are explicitly marked `NOT_AVAILABLE_without_GT_matching` rather than inferred.",
        "- `BAD_IOU_OR_LABEL_MISMATCH` means predictions exist but AP remains zero, so the next check is label alignment plus IoU distribution.",
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

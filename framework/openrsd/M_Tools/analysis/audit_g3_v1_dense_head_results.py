#!/usr/bin/env python
"""Audit existing HRRSD G3-v1 dense-head consistency results.

This script does not launch training or inference. It reads already generated
HRRSD G3/density/no-G3 eval JSONs and deployment-risk CSVs, then applies the
strict reviewer rule: G3-v1 only counts as a positive method result if it beats
matched no-G3 controls, not merely the epoch3 baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_EVAL_ROOT = Path("work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619")
DEFAULT_RISK_ROOT = Path("work_dirs/gs3c_sise_problem_reframing_20260619")
DEFAULT_OUT_DIR = Path("work_dirs/semantic_scale_six_experiments_20260620")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_g3_v1_negative_partial_audit.md")


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_eval_metric(eval_dir):
    eval_dir = Path(eval_dir)
    candidates = []
    if eval_dir.exists():
        candidates.extend(sorted(eval_dir.glob("*/20*.json")))
        candidates.extend(sorted(eval_dir.glob("*.json")))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            payload = {**metrics, **payload}
        mAP = payload.get("dota/mAP", payload.get("mAP"))
        ap50 = payload.get("dota/AP50", payload.get("AP50"))
        if mAP is None:
            mAP = payload.get("coco/bbox_mAP")
        if ap50 is None:
            ap50 = payload.get("coco/bbox_mAP_50")
        if mAP is not None:
            return {
                "eval_json": str(path),
                "mAP": float(mAP),
                "AP50": float(ap50) if ap50 is not None else 0.0,
            }
    return {"eval_json": "", "mAP": 0.0, "AP50": 0.0}


def infer_kind(exp_id, variant):
    raw = f"{exp_id} {variant}".lower()
    if raw.startswith("nog3") or "nog3" in raw:
        return "nog3"
    if raw.startswith("density") or "density" in raw:
        return "density"
    if raw.startswith("g3") or "g3" in raw:
        return "g3"
    return "other"


def eval_dir_from_prediction(prediction_path, eval_root):
    pred = Path(prediction_path)
    if pred.is_absolute():
        return pred.parent
    if pred.parts and pred.parts[0] == str(eval_root).split("/")[0]:
        return pred.parent
    if str(pred).startswith(str(eval_root)):
        return pred.parent
    return Path(pred).parent


def build_variant_rows(eval_root, risk_root, baseline_metric):
    rows = []
    for csv_path in sorted(Path(risk_root).glob(
            "hrrsd_*/deployment_risk_variant_summary.csv")):
        exp_id = csv_path.parent.name.replace("hrrsd_", "")
        kind_hint = infer_kind(exp_id, "")
        if kind_hint not in {"g3", "density", "nog3"}:
            continue
        data = read_csv_rows(csv_path)
        if len(data) < 2:
            continue
        base, method = data[0], data[1]
        eval_dir = eval_dir_from_prediction(
            method.get("predictions", ""), eval_root)
        metric = read_eval_metric(eval_dir)
        kind = infer_kind(exp_id, method.get("variant", ""))
        row = {
            "exp_id": exp_id,
            "kind": kind,
            "variant": method.get("variant", ""),
            "mAP": metric["mAP"],
            "AP50": metric["AP50"],
            "mAP_delta_vs_epoch3_baseline": (
                metric["mAP"] - baseline_metric["mAP"]),
            "localized_correct_delta": (
                to_float(method.get("localized_correct"))
                - to_float(base.get("localized_correct"))),
            "localized_precision_delta": (
                to_float(method.get("localized_precision"))
                - to_float(base.get("localized_precision"))),
            "sise_logz_total_delta": (
                to_float(method.get("sise_logz_wrong_excl_sibling"))
                - to_float(base.get("sise_logz_wrong_excl_sibling"))),
            "sise_p0199_total_delta": (
                to_float(method.get("sise_p0199_wrong_excl_sibling"))
                - to_float(base.get("sise_p0199_wrong_excl_sibling"))),
            "score05_wrong_delta": (
                to_float(method.get(
                    "wrong_excl_sibling_ge_thr_score_ge_0p5"))
                - to_float(base.get(
                    "wrong_excl_sibling_ge_thr_score_ge_0p5"))),
            "score05_sise_logz_delta": (
                to_float(method.get("sise_logz_ge_thr_score_ge_0p5"))
                - to_float(base.get("sise_logz_ge_thr_score_ge_0p5"))),
            "top5000_correct_delta": (
                to_float(method.get("correct_topk_top5000"))
                - to_float(base.get("correct_topk_top5000"))),
            "top5000_sise_logz_delta": (
                to_float(method.get("sise_logz_topk_top5000"))
                - to_float(base.get("sise_logz_topk_top5000"))),
            "top5000_sise_p0199_delta": (
                to_float(method.get("sise_p0199_topk_top5000"))
                - to_float(base.get("sise_p0199_topk_top5000"))),
            "eval_json": metric["eval_json"],
            "risk_csv": str(csv_path),
            "predictions": method.get("predictions", ""),
        }
        rows.append(row)
    return rows


def best_by(rows, key, reverse=True):
    selected = [row for row in rows if row.get(key) is not None]
    if not selected:
        return {}
    return sorted(selected, key=lambda row: row[key], reverse=reverse)[0]


def md_table(rows, columns, max_rows=12):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:max_rows]:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if "delta" in col or col in {"mAP", "AP50"}:
                    values.append(f"{value:.6f}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path, review, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_map = sorted(rows, key=lambda row: row["mAP"], reverse=True)
    by_sise = sorted(rows, key=lambda row: row["sise_logz_total_delta"])
    lines = [
        "# G3-v1 Dense-Head Consistency Existing Result Audit - 2026-06-20",
        "",
        "本文件只整理已有 HRRSD G3/density/no-G3 训练结果，不启动新 GPU "
        "实验。审稿口径按用户确认的 gate：G3-v1 必须超过 matched no-G3 "
        "control，不能只和 epoch3 baseline 比。",
        "",
        "## Verdict",
        "",
        f"- status: `{review['status']}`",
        f"- baseline mAP: `{review['baseline']['mAP']:.6f}`",
        f"- best no-G3 mAP: `{review['best_nog3'].get('mAP', 0.0):.6f}` "
        f"(`{review['best_nog3'].get('exp_id', '')}`)",
        f"- best G3 mAP: `{review['best_g3_by_map'].get('mAP', 0.0):.6f}` "
        f"(`{review['best_g3_by_map'].get('exp_id', '')}`)",
        f"- best G3 beats best no-G3 by mAP: "
        f"`{review['best_g3_beats_best_nog3_map']}`",
        f"- best G3 SISE reduction exp: "
        f"`{review['best_g3_by_sise_reduction'].get('exp_id', '')}` "
        f"(`delta={review['best_g3_by_sise_reduction'].get('sise_logz_total_delta', 0.0):.0f}`)",
        "",
        "严格结论：`G3-v1 = negative/partial`。它证明 dense-head 训练路径"
        "可跑，部分配置能改变 semantic-scale risk/precision，但目前没有证明"
        "超过同 epoch/same protocol 的 no-G3 继续训练控制。",
        "",
        "## Top By mAP",
        "",
        md_table(by_map, [
            "exp_id", "kind", "mAP", "mAP_delta_vs_epoch3_baseline",
            "sise_logz_total_delta", "top5000_correct_delta",
            "top5000_sise_logz_delta",
        ]),
        "",
        "## Top By SISE Reduction",
        "",
        md_table(by_sise, [
            "exp_id", "kind", "mAP", "mAP_delta_vs_epoch3_baseline",
            "sise_logz_total_delta", "sise_p0199_total_delta",
            "localized_precision_delta",
        ]),
        "",
        "## G3-v2 Gate",
        "",
        "后续 G3-v2 只有在以下条件满足后才应启动 GPU：",
        "",
        "- 使用 matched no-G3 same-seed/same-epoch control。",
        "- 主成功标准是超过 matched no-G3 的 `mAP` 或 AP-sensitive utility，"
        "不是超过 epoch3 baseline。",
        "- 同时报告 `SISE`, `rank-tail`, `classwise AP`，防止只压低错误分数"
        "但伤害排序。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_g3_v1_results(eval_root=DEFAULT_EVAL_ROOT,
                         risk_root=DEFAULT_RISK_ROOT,
                         out_dir=DEFAULT_OUT_DIR,
                         result_md=DEFAULT_RESULT_MD):
    eval_root = Path(eval_root)
    risk_root = Path(risk_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_eval_metric(eval_root / "eval_epoch3_baseline_full")
    rows = build_variant_rows(eval_root, risk_root, baseline)
    rows = sorted(rows, key=lambda row: row["mAP"], reverse=True)
    g3_rows = [row for row in rows if row["kind"] == "g3"]
    nog3_rows = [row for row in rows if row["kind"] == "nog3"]
    density_rows = [row for row in rows if row["kind"] == "density"]

    best_nog3 = best_by(nog3_rows, "mAP")
    best_g3_by_map = best_by(g3_rows, "mAP")
    best_density = best_by(density_rows, "mAP")
    best_g3_by_sise = best_by(g3_rows, "sise_logz_total_delta", reverse=False)
    best_g3_beats = bool(
        best_g3_by_map and best_nog3
        and best_g3_by_map["mAP"] > best_nog3["mAP"])
    if not g3_rows:
        status = "pending_no_g3_results"
    elif best_g3_beats:
        status = "positive"
    elif best_g3_by_sise and best_g3_by_sise["sise_logz_total_delta"] < 0:
        status = "negative_partial"
    else:
        status = "negative"

    csv_path = out_dir / "g3_v1_dense_head_audit.csv"
    json_path = out_dir / "g3_v1_dense_head_audit.json"
    write_csv(csv_path, rows)
    review = {
        "status": status,
        "eval_root": str(eval_root),
        "risk_root": str(risk_root),
        "result_md": str(result_md),
        "csv_path": str(csv_path),
        "baseline": baseline,
        "num_rows": len(rows),
        "num_g3": len(g3_rows),
        "num_density": len(density_rows),
        "num_nog3": len(nog3_rows),
        "best_nog3": best_nog3,
        "best_g3_by_map": best_g3_by_map,
        "best_density_by_map": best_density,
        "best_g3_by_sise_reduction": best_g3_by_sise,
        "best_g3_beats_best_nog3_map": best_g3_beats,
    }
    json_path.write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(result_md, review, rows)
    return review


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit existing HRRSD G3-v1 dense-head results.")
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--risk-root", default=str(DEFAULT_RISK_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    return parser.parse_args()


def main():
    args = parse_args()
    review = audit_g3_v1_results(
        eval_root=args.eval_root,
        risk_root=args.risk_root,
        out_dir=args.out_dir,
        result_md=args.result_md)
    print(json.dumps({
        "status": review["status"],
        "best_g3_beats_best_nog3_map": (
            review["best_g3_beats_best_nog3_map"]),
        "result_md": review["result_md"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

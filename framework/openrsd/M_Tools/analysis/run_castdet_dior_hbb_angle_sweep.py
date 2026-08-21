#!/usr/bin/env python
"""Run G1 HBB scale-prior audit across CastDet DIOR angle-sweep predictions."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from apply_scale_prior_to_predictions import apply_scale_prior_to_predictions
from eval_hbb_predictions_coco_json import evaluate as coco_evaluate
from evaluate_sise_hbb_coco import (
    build_delta,
    load_coco_gt,
    load_priors,
    parse_thresholds,
    summarize_variant,
)


DEFAULT_DIOR_CLASSES = (
    "airplane",
    "baseballfield",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "airport",
    "basketballcourt",
    "groundtrackfield",
    "windmill",
)


def parse_class_names(raw):
    if not raw:
        return list(DEFAULT_DIOR_CLASSES)
    return [item.strip() for item in raw.split(",") if item.strip()]


def angle_dirs(castdet_root):
    for path in sorted(Path(castdet_root).glob("angle_*")):
        if not path.is_dir():
            continue
        pred_path = path / "predictions.pkl"
        if pred_path.exists():
            yield path.name, pred_path


def summarize_angle(angle, pred_path, ann_json, prior_csv, out_dir, class_names,
                    args, priors):
    angle_out = out_dir / angle
    angle_out.mkdir(parents=True, exist_ok=True)
    g1_pred_path = angle_out / f"{angle}_g1_z{args.z_margin:g}_l{args.downweight_lambda:g}_predictions.pkl"
    g1_summary_path = angle_out / f"{angle}_g1_summary.json"

    g1_apply_summary = apply_scale_prior_to_predictions(
        input_pkl=pred_path,
        output_pkl=g1_pred_path,
        class_area_priors_csv=prior_csv,
        class_names=class_names,
        strategy="log_area_calibrate",
        log_area_z_margin=args.z_margin,
        downweight_lambda=args.downweight_lambda,
    )
    g1_summary_path.write_text(
        json.dumps(g1_apply_summary, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")

    base_eval = coco_evaluate(ann_json, pred_path)
    g1_eval = coco_evaluate(ann_json, g1_pred_path)
    (angle_out / f"{angle}_base_coco_eval.json").write_text(
        json.dumps(base_eval, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    (angle_out / f"{angle}_g1_coco_eval.json").write_text(
        json.dumps(g1_eval, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")

    gt = load_coco_gt(ann_json)
    sise_args = SimpleNamespace(
        iou_thr=args.iou_thr,
        z_thr=args.sise_z_thr,
        scale_ratio_thr=args.scale_ratio_thr,
        max_dets_per_img=args.max_dets_per_img,
        max_images=0,
    )
    score_thrs = parse_thresholds(args.score_thrs)
    base_sise = summarize_variant(
        "base", Path(pred_path), gt, priors, sise_args, score_thrs, set())
    g1_sise = summarize_variant(
        "g1", g1_pred_path, gt, priors, sise_args, score_thrs, set())
    sise_delta = build_delta(base_sise, g1_sise)
    sise_payload = {
        "angle": angle,
        "ann_json": str(ann_json),
        "base_predictions": str(pred_path),
        "g1_predictions": str(g1_pred_path),
        "variants": [base_sise, g1_sise],
        "deltas": [sise_delta],
    }
    (angle_out / f"{angle}_sise_summary.json").write_text(
        json.dumps(sise_payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")

    base_metrics = base_eval["metrics"]
    g1_metrics = g1_eval["metrics"]
    row = {
        "angle": angle,
        "ann_json": str(ann_json),
        "base_predictions": str(pred_path),
        "g1_predictions": str(g1_pred_path),
        "detections_calibrated": g1_apply_summary["detections_calibrated"],
        "calibrate_rate": g1_apply_summary["calibrate_rate"],
        "base_mAP": base_metrics["coco/bbox_mAP"],
        "g1_mAP": g1_metrics["coco/bbox_mAP"],
        "ap_delta": g1_metrics["coco/bbox_mAP"] - base_metrics["coco/bbox_mAP"],
        "base_AP50": base_metrics["coco/bbox_mAP_50"],
        "g1_AP50": g1_metrics["coco/bbox_mAP_50"],
        "AP50_delta": g1_metrics["coco/bbox_mAP_50"] - base_metrics["coco/bbox_mAP_50"],
        "base_wrong_score_ge_0p5": base_sise["wrong_excl_sibling_score_ge_0p5"],
        "g1_wrong_score_ge_0p5": g1_sise["wrong_excl_sibling_score_ge_0p5"],
        "wrong_score_ge_0p5_delta": sise_delta["wrong_excl_sibling_score_ge_0p5_delta"],
        "base_wrong_score_ge_0p9": base_sise["wrong_excl_sibling_score_ge_0p9"],
        "g1_wrong_score_ge_0p9": g1_sise["wrong_excl_sibling_score_ge_0p9"],
        "wrong_score_ge_0p9_delta": sise_delta["wrong_excl_sibling_score_ge_0p9_delta"],
        "base_sise_logz_score_ge_0p5": base_sise["sise_logz_score_ge_0p5"],
        "g1_sise_logz_score_ge_0p5": g1_sise["sise_logz_score_ge_0p5"],
        "sise_logz_score_ge_0p5_delta": sise_delta["sise_logz_score_ge_0p5_delta"],
        "base_sise_p0199_score_ge_0p5": base_sise["sise_p0199_score_ge_0p5"],
        "g1_sise_p0199_score_ge_0p5": g1_sise["sise_p0199_score_ge_0p5"],
        "sise_p0199_score_ge_0p5_delta": sise_delta["sise_p0199_score_ge_0p5_delta"],
    }
    return row


def aggregate(rows):
    if not rows:
        return {}
    return {
        "angle_count": len(rows),
        "ap_non_regression_count": sum(row["ap_delta"] >= 0 for row in rows),
        "ap_delta_mean": sum(row["ap_delta"] for row in rows) / len(rows),
        "ap_delta_min": min(row["ap_delta"] for row in rows),
        "ap_delta_max": max(row["ap_delta"] for row in rows),
        "AP50_delta_mean": sum(row["AP50_delta"] for row in rows) / len(rows),
        "wrong_score_ge_0p5_delta_sum": sum(
            row["wrong_score_ge_0p5_delta"] for row in rows),
        "wrong_score_ge_0p9_delta_sum": sum(
            row["wrong_score_ge_0p9_delta"] for row in rows),
        "sise_logz_score_ge_0p5_delta_sum": sum(
            row["sise_logz_score_ge_0p5_delta"] for row in rows),
        "sise_p0199_score_ge_0p5_delta_sum": sum(
            row["sise_p0199_score_ge_0p5_delta"] for row in rows),
    }


def write_markdown(path, payload):
    agg = payload["aggregate"]
    lines = [
        "# CastDet DIOR-HBB 12-Angle G1 Audit",
        "",
        f"- castdet_root: `{payload['castdet_root']}`",
        f"- ann_root: `{payload['ann_root']}`",
        f"- prior_csv: `{payload['class_area_priors_csv']}`",
        f"- G1: `z={payload['z_margin']}`, `lambda={payload['downweight_lambda']}`",
        "",
        "## Aggregate",
        "",
        f"- angle_count: `{agg.get('angle_count', 0)}`",
        f"- AP non-regression: `{agg.get('ap_non_regression_count', 0)}/{agg.get('angle_count', 0)}`",
        f"- mean ap_delta: `{agg.get('ap_delta_mean', 0.0)}`",
        f"- min ap_delta: `{agg.get('ap_delta_min', 0.0)}`",
        f"- wrong@0.5 delta sum: `{agg.get('wrong_score_ge_0p5_delta_sum', 0)}`",
        f"- wrong@0.9 delta sum: `{agg.get('wrong_score_ge_0p9_delta_sum', 0)}`",
        f"- SISE_logz@0.5 delta sum: `{agg.get('sise_logz_score_ge_0p5_delta_sum', 0)}`",
        f"- SISE_p0199@0.5 delta sum: `{agg.get('sise_p0199_score_ge_0p5_delta_sum', 0)}`",
        "",
        "## Per Angle",
        "",
        "| angle | base_mAP | g1_mAP | ap_delta | base_AP50 | g1_AP50 | AP50_delta | wrong@0.5 delta | wrong@0.9 delta | SISE_logz@0.5 delta | SISE_p0199@0.5 delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['angle']} | {row['base_mAP']:.10f} | {row['g1_mAP']:.10f} | "
            f"{row['ap_delta']:.10f} | {row['base_AP50']:.10f} | "
            f"{row['g1_AP50']:.10f} | {row['AP50_delta']:.10f} | "
            f"{row['wrong_score_ge_0p5_delta']} | {row['wrong_score_ge_0p9_delta']} | "
            f"{row['sise_logz_score_ge_0p5_delta']} | "
            f"{row['sise_p0199_score_ge_0p5_delta']} |")
    Path(path).write_text("\n".join(lines) + os.linesep, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--castdet-root", required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--class-area-priors-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-names", default=",".join(DEFAULT_DIOR_CLASSES))
    parser.add_argument("--z-margin", type=float, default=2.0)
    parser.add_argument("--downweight-lambda", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--sise-z-thr", type=float, default=4.0)
    parser.add_argument("--scale-ratio-thr", type=float, default=4.0)
    parser.add_argument("--score-thrs", default="0.5,0.7,0.9,0.99,0.999")
    parser.add_argument("--max-dets-per-img", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = parse_class_names(args.class_names)
    priors = load_priors(args.class_area_priors_csv)

    rows = []
    for angle, pred_path in angle_dirs(args.castdet_root):
        ann_json = Path(args.ann_root) / angle / "annotations.json"
        if not ann_json.exists():
            continue
        rows.append(summarize_angle(
            angle=angle,
            pred_path=pred_path,
            ann_json=ann_json,
            prior_csv=args.class_area_priors_csv,
            out_dir=out_dir,
            class_names=class_names,
            args=args,
            priors=priors,
        ))

    payload = {
        "castdet_root": args.castdet_root,
        "ann_root": args.ann_root,
        "class_area_priors_csv": args.class_area_priors_csv,
        "z_margin": args.z_margin,
        "downweight_lambda": args.downweight_lambda,
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    summary_json = out_dir / "castdet_dior_hbb_angle_sweep_summary.json"
    summary_md = out_dir / "castdet_dior_hbb_angle_sweep_summary.md"
    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep,
        encoding="utf-8")
    write_markdown(summary_md, payload)
    print(json.dumps({
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "aggregate": payload["aggregate"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze FOCUS-T-Safe shadow text scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_tsafe_common import (
    EXP_DIR,
    md_table,
    read_csv,
    resolve,
    roc_auc,
    safe_float,
    summarize,
    write_csv,
    write_json,
)


def decide_shadow_status(
        auc: float | None,
        degenerate_high_rate: float,
        padding_high_rate: float,
        high_rate_limit: float = 0.30) -> dict[str, object]:
    if degenerate_high_rate > high_rate_limit or padding_high_rate > high_rate_limit:
        return {
            "status": "SHADOW_UNSAFE_AS_GATE",
            "reason": "degenerate_or_padding_high_score_rate",
        }
    if auc is None:
        return {"status": "SHADOW_NO_SIGNAL", "reason": "missing_auc"}
    if auc >= 0.65:
        return {"status": "SHADOW_SIGNAL_PASS", "reason": "auc_pass"}
    if auc < 0.55:
        return {"status": "SHADOW_NO_SIGNAL", "reason": "auc_below_0.55"}
    return {"status": "SHADOW_INDETERMINATE", "reason": "auc_mid_band"}


def combined_score(row: Mapping[str, Any]) -> float:
    return (
        safe_float(row.get("eqtext_sv_similarity")) +
        safe_float(row.get("negative_text_margin")) +
        safe_float(row.get("visual_text_consistency"))
    ) / 3.0


def high_rate(rows: list[dict[str, str]], category: str,
              score_key: str = "combined_score",
              threshold: float = 0.5) -> float:
    selected = [row for row in rows if row.get("audit_category") == category]
    if not selected:
        return 0.0
    return sum(safe_float(row.get(score_key)) >= threshold for row in selected) / len(selected)


def write_figures(rows: list[dict[str, str]], output_dir: Path) -> dict[str, str]:
    paths = {
        "roc_png": str(output_dir / "figures" / "tsafe_shadow_roc.png"),
        "roc_pdf": str(output_dir / "figures" / "tsafe_shadow_roc.pdf"),
        "dist_png": str(output_dir / "figures" / "tsafe_shadow_score_distribution.png"),
        "dist_pdf": str(output_dir / "figures" / "tsafe_shadow_score_distribution.pdf"),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [int(row["binary_true_vehicle"]) for row in rows
                  if row.get("binary_true_vehicle") in {"0", "1", 0, 1}]
        scores = [safe_float(row.get("combined_score")) for row in rows
                  if row.get("binary_true_vehicle") in {"0", "1", 0, 1}]
        order = sorted(zip(scores, labels), reverse=True)
        positives = max(1, sum(labels))
        negatives = max(1, len(labels) - sum(labels))
        tpr = [0.0]
        fpr = [0.0]
        tp = fp = 0
        for _score, label in order:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label="combined")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths["roc_png"])
        plt.savefig(paths["roc_pdf"])
        plt.close()

        by_cat: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_cat[row.get("audit_category", "unknown")].append(
                safe_float(row.get("combined_score")))
        plt.figure(figsize=(7, 4))
        labels_plot = sorted(by_cat)
        data = [by_cat[label] for label in labels_plot]
        plt.boxplot(data, labels=labels_plot, vert=True)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Combined shadow score")
        plt.tight_layout()
        plt.savefig(paths["dist_png"])
        plt.savefig(paths["dist_pdf"])
        plt.close()
    except Exception as exc:
        for path in paths.values():
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"Figure unavailable: {exc}\n", encoding="utf-8")
    return paths


def write_report(path: Path, payload: dict[str, Any],
                 auc_rows: list[dict[str, Any]],
                 means_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# FOCUS-T-Safe Shadow Signal Report",
        "",
        f"- status: `{payload['status']}`",
        f"- reason: `{payload['reason']}`",
        f"- combined_auc: `{payload.get('combined_auc')}`",
        f"- degenerate_high_rate: `{payload['degenerate_high_rate']}`",
        f"- padding_high_rate: `{payload['padding_high_rate']}`",
        f"- text_delta_norm_max: `{payload['text_delta_norm_max']}`",
        f"- text_interclass_cos_collapse_flag: `{payload['text_interclass_cos_collapse_flag']}`",
        "",
        "## AUC",
        "",
    ]
    lines.extend(md_table(auc_rows, ["metric", "auc", "n"]))
    lines.extend(["", "## Category Means", ""])
    lines.extend(md_table(
        means_rows,
        ["audit_category", "count", "eqtext_sv_similarity_mean",
         "negative_text_margin_mean", "visual_text_consistency_mean",
         "combined_score_mean"]))
    lines.extend([
        "",
        "No detector logits, NMS, support bank, or checkpoint weights were modified.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=EXP_DIR / "shadow_scores" / "tsafe_shadow_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    scores = args.scores
    output_dir = args.output_dir
    if not scores.is_absolute():
        scores = Path.cwd() / scores
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    rows = read_csv(scores)
    for row in rows:
        row["combined_score"] = combined_score(row)

    labeled = [row for row in rows if str(row.get("binary_true_vehicle")) in {"0", "1"}]
    labels = [int(row["binary_true_vehicle"]) for row in labeled]
    metrics = [
        "eqtext_sv_similarity",
        "negative_text_margin",
        "visual_text_consistency",
        "combined_score",
    ]
    auc_rows = []
    for metric in metrics:
        auc_rows.append({
            "metric": metric,
            "auc": roc_auc(labels, [safe_float(row.get(metric)) for row in labeled]),
            "n": len(labeled),
        })
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("audit_category", "unknown"))].append(row)
    means_rows: list[dict[str, Any]] = []
    for category in sorted(by_cat):
        cat_rows = by_cat[category]
        means_rows.append({
            "audit_category": category,
            "count": len(cat_rows),
            "eqtext_sv_similarity_mean": summarize(
                [safe_float(row.get("eqtext_sv_similarity")) for row in cat_rows])["mean"],
            "negative_text_margin_mean": summarize(
                [safe_float(row.get("negative_text_margin")) for row in cat_rows])["mean"],
            "visual_text_consistency_mean": summarize(
                [safe_float(row.get("visual_text_consistency")) for row in cat_rows])["mean"],
            "combined_score_mean": summarize(
                [safe_float(row.get("combined_score")) for row in cat_rows])["mean"],
        })
    degenerate_high = high_rate(rows, "degenerate_large_sv_box")
    padding_high = high_rate(rows, "padding_artifact")
    combined_auc = next(
        (row["auc"] for row in auc_rows if row["metric"] == "combined_score"),
        None)
    decision = decide_shadow_status(combined_auc, degenerate_high, padding_high)
    payload = {
        **decision,
        "scores": str(scores),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "combined_auc": combined_auc,
        "degenerate_high_rate": degenerate_high,
        "padding_high_rate": padding_high,
        "text_delta_norm_max": max(
            [safe_float(row.get("text_delta_norm")) for row in rows] or [0.0]),
        "text_interclass_cos_max": max(
            [safe_float(row.get("text_interclass_cos_max")) for row in rows] or [0.0]),
        "text_interclass_cos_collapse_flag": max(
            [safe_float(row.get("text_interclass_cos_max")) for row in rows] or [0.0]) >= 0.90,
        "text_branch_final_logit_effect": False,
        "support_bank_replaced": False,
    }
    fig_paths = write_figures(rows, output_dir)
    payload.update(fig_paths)
    write_csv(output_dir / "tables" / "tsafe_shadow_auc.csv", auc_rows)
    write_csv(output_dir / "tables" / "tsafe_shadow_category_means.csv", means_rows)
    write_json(output_dir / "reports" / "tsafe_shadow_signal_report.json", payload)
    write_report(output_dir / "reports" / "tsafe_shadow_signal_report.md",
                 payload, auc_rows, means_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

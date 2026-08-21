#!/usr/bin/env python3
"""Analyze verified degenerate large small-vehicle boxes as a separate failure mode."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    use = rows[:limit] if limit else rows
    if headers is None:
        headers = list(use[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in use:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except Exception:
        return 0.0


def save_fig(path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-summary-dir", required=True)
    args = parser.parse_args()
    root = Path(args.visual_summary_dir)
    audit_dir = root / "audit"
    fig_dir = root / "figures"
    pool = read_csv(audit_dir / "verified_expanded_candidate_pool.csv")
    if not pool:
        pool = read_csv(audit_dir / "urgent_crop_pack_integrity_audit.csv")
    deg = [r for r in pool if r.get("audit_category") == "degenerate_large_sv_box" or r.get("is_oversized") in {"True", "true"} or r.get("is_oversized_for_small_vehicle") == "true"]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        groups[(row.get("model_name", ""), str(row.get("angle", "")))].append(row)
    analysis = []
    for (model, angle), rows in sorted(groups.items()):
        deg_rows = [r for r in rows if r in deg or r.get("audit_category") == "degenerate_large_sv_box"]
        if not rows:
            continue
        scores = [f(r, "score") for r in deg_rows]
        areas = [f(r, "raw_box_area") for r in deg_rows]
        valid = [f(r, "valid_mask_ratio_inside_box") for r in deg_rows]
        pad = [f(r, "padding_overlap_ratio") for r in deg_rows]
        tiles = Counter(r.get("tile_id", "") for r in deg_rows)
        analysis.append({
            "model_name": model,
            "angle": angle,
            "num_sv_pred": len(rows),
            "num_degenerate_large_sv": len(deg_rows),
            "degenerate_large_sv_ratio": f"{len(deg_rows) / max(len(rows), 1):.6f}",
            "mean_score": f"{sum(scores) / len(scores):.6f}" if scores else "",
            "median_score": f"{median(scores):.6f}" if scores else "",
            "score_p90": f"{sorted(scores)[int(0.9 * (len(scores)-1))]:.6f}" if scores else "",
            "mean_area": f"{sum(areas) / len(areas):.6f}" if areas else "",
            "area_vs_sv_gt_p995": "see sv_gt_size_distribution.csv",
            "padding_overlap_mean": f"{sum(pad) / len(pad):.6f}" if pad else "",
            "valid_mask_ratio_mean": f"{sum(valid) / len(valid):.6f}" if valid else "",
            "top_tiles": ";".join(f"{k}:{v}" for k, v in tiles.most_common(5)),
        })
    write_csv(audit_dir / "degenerate_large_sv_box_analysis.csv", analysis)

    by_model = defaultdict(lambda: {"num_sv_pred": 0, "num_degenerate_large_sv": 0})
    by_angle = defaultdict(lambda: {"num_sv_pred": 0, "num_degenerate_large_sv": 0})
    for row in analysis:
        m = row["model_name"]
        a = row["angle"]
        by_model[m]["num_sv_pred"] += int(row["num_sv_pred"])
        by_model[m]["num_degenerate_large_sv"] += int(row["num_degenerate_large_sv"])
        by_angle[a]["num_sv_pred"] += int(row["num_sv_pred"])
        by_angle[a]["num_degenerate_large_sv"] += int(row["num_degenerate_large_sv"])

    if by_model:
        labels = sorted(by_model)
        vals = [by_model[x]["num_degenerate_large_sv"] for x in labels]
        plt.figure(figsize=(max(8, len(labels) * 0.7), 4))
        plt.bar(labels, vals, color="#b23a48")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("degenerate large SV count")
        plt.title("Degenerate large SV boxes by model")
        save_fig(fig_dir / "fig_degenerate_large_sv_by_model")
    if by_angle:
        labels = sorted(by_angle, key=lambda x: int(float(x)))
        vals = [by_angle[x]["num_degenerate_large_sv"] for x in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, vals, color="#335c67")
        plt.xlabel("angle")
        plt.ylabel("degenerate large SV count")
        plt.title("Degenerate large SV boxes by angle")
        save_fig(fig_dir / "fig_degenerate_large_sv_by_angle")
    scores = [f(r, "score") for r in deg]
    if scores:
        plt.figure(figsize=(7, 4))
        plt.hist(scores, bins=20, color="#6c757d", edgecolor="white")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.title("Degenerate large SV score distribution")
        save_fig(fig_dir / "fig_degenerate_large_sv_score_distribution")

    model_totals = [
        {
            "model_name": m,
            "num_sv_pred": v["num_sv_pred"],
            "num_degenerate_large_sv": v["num_degenerate_large_sv"],
            "ratio": f"{v['num_degenerate_large_sv'] / max(v['num_sv_pred'], 1):.6f}",
        }
        for m, v in sorted(by_model.items(), key=lambda kv: kv[1]["num_degenerate_large_sv"], reverse=True)
    ]
    md = [
        "# Degenerate Large Small-Vehicle Box Analysis",
        "",
        "Scope: verified/mined full-prediction-pool candidates used for expanded crop generation. These boxes are analyzed as a separate failure mode and are not used for missing-annotation corrected-FSV.",
        "",
        "## Answers",
        "",
        f"1. r3det_kfiou produces many degenerate large SV boxes in the mined pool: `{next((r['num_degenerate_large_sv'] for r in model_totals if r['model_name']=='r3det_kfiou'), 0)}`.",
        "2. Angle concentration is shown in `fig_degenerate_large_sv_by_angle`.",
        "3. Score distribution is shown in `fig_degenerate_large_sv_score_distribution`.",
        "4. Padding overlap is reported per model/angle; padding artifacts are separate and excluded from corrected-FSV.",
        "5. These should be reported separately from unannotated true-SV contamination.",
        "",
        "## By Model",
        "",
        md_table(model_totals),
        "",
        "## By Model/Angle",
        "",
        md_table(analysis, limit=80),
        "",
        "Allowed wording: `Some closed-set models produce verified degenerate large small-vehicle predictions under rotation.`",
        "",
        "Forbidden wording: `These are unannotated vehicles`; `These prove hallucination`; `These should be included in corrected-FSV missing-annotation audit.`",
    ]
    write_md(audit_dir / "degenerate_large_sv_box_analysis.md", "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

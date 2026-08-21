#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _model_false_hub_table(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: _float(r, "mean_final_fsv"), reverse=True)
    return _table(
        [
            "Model",
            "Status",
            "Mean FSV",
            "BG-FSV",
            "FR-SV",
            "True SV Recall",
            "Rows",
        ],
        [
            [
                row["model_name"],
                row.get("status", ""),
                _fmt(_float(row, "mean_final_fsv")),
                _fmt(_float(row, "mean_bg_fsv")),
                _fmt(_float(row, "mean_fr_sv")),
                _fmt(_float(row, "mean_true_sv_recall")),
                row.get("num_rows", ""),
            ]
            for row in ordered
        ],
    )


def _stage_table(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: _float(r, "mean_post_nms_fr_sv"), reverse=True)
    return _table(
        [
            "Model",
            "Family",
            "Arch",
            "Mean Post-NMS FR-SV",
            "Dense",
            "Pre-NMS",
            "Post-NMS",
            "Status",
        ],
        [
            [
                row["model_name"],
                row.get("model_family", ""),
                row.get("architecture_type", ""),
                _fmt(_float(row, "mean_post_nms_fr_sv")),
                row.get("dense_logits_status", ""),
                row.get("pre_nms_status", ""),
                row.get("post_nms_status", ""),
                row.get("status", ""),
            ]
            for row in ordered
        ],
    )


def _rotation_summary(rows: list[dict]) -> tuple[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)
    summary = []
    for model, items in sorted(grouped.items()):
        worst_angles = Counter(int(float(row.get("worst_angle_sv") or 0)) for row in items)
        mode_angle, mode_count = worst_angles.most_common(1)[0]
        summary.append(
            {
                "model_name": model,
                "mean_rg_sv": _mean([_float(row, "rg_sv") for row in items]),
                "mean_arg_sv": _mean([_float(row, "arg_sv") for row in items]),
                "worst_angle_mode": mode_angle,
                "worst_angle_mode_count": mode_count,
                "num_tiles": len(items),
            }
        )
    ordered = sorted(summary, key=lambda r: r["mean_rg_sv"], reverse=True)
    table = _table(
        ["Model", "Mean RG-SV", "Mean ARG-SV", "Most Frequent Worst Angle", "Count", "Tiles"],
        [
            [
                row["model_name"],
                _fmt(row["mean_rg_sv"]),
                _fmt(row["mean_arg_sv"]),
                str(row["worst_angle_mode"]),
                str(row["worst_angle_mode_count"]),
                str(row["num_tiles"]),
            ]
            for row in ordered
        ],
    )
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle",
    )
    parser.add_argument(
        "--output",
        default="resultmd/exp_rotation_semantic_attractor_full_closedset/fres_full_closedset_s2_12angle.md",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = run_dir / "metrics"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(run_dir / "manifest.json")
    status_rows = _read_csv(metrics / "run_status_by_model.csv")
    false_hub_rows = _read_csv(metrics / "full_closedset_false_hub_summary_by_model.csv")
    stage_rows = _read_csv(metrics / "stage_decomposition_summary.csv")
    rotation_rows = _read_csv(metrics / "full_closedset_rotation_gain.csv")
    concentration_rows = _read_csv(metrics / "full_closedset_concentration.csv")
    concentration = concentration_rows[0] if concentration_rows else {}
    rotation_table, _ = _rotation_summary(rotation_rows)

    completed = sum(1 for row in status_rows if row.get("status") == "DONE_FULL")
    expected_tile_angles = sum(int(row.get("expected_tile_angle") or 0) for row in status_rows)
    failed_tile_angles = sum(int(row.get("failed_tile_angle") or 0) for row in status_rows)
    raw_completed = sum(int(row.get("raw_completed_tile_angle") or 0) for row in status_rows)
    canonical_completed = sum(int(row.get("canonical_completed_tile_angle") or 0) for row in status_rows)
    models = manifest.get("selected_models", [])
    angles = manifest.get("angles", [])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_model = min(false_hub_rows, key=lambda r: _float(r, "mean_final_fsv"))
    worst_model = max(false_hub_rows, key=lambda r: _float(r, "mean_final_fsv"))

    sections = [
        "# Rotation Semantic Attractor: Full Closed-Set 12-Angle Summary",
        "",
        f"- generated_at: `{generated_at}`",
        f"- run_dir: `{run_dir}`",
        f"- split: `{manifest.get('split_name', '')}`",
        f"- angles: `{','.join(str(a) for a in angles)}`",
        f"- models: `{len(models)}`",
        f"- seed: `{manifest.get('args', {}).get('seed', '')}`",
        f"- IoU threshold: `{manifest.get('args', {}).get('iou_thr', '')}`",
        "",
        "## Executive Summary",
        "",
        f"The full closed-set inference run is complete: `{completed}/{len(status_rows)}` models are `DONE_FULL`, with `{raw_completed}` raw and `{canonical_completed}` canonical tile-angle outputs out of `{expected_tile_angles}` expected per output type, and `{failed_tile_angles}` failed tile-angles.",
        "",
        f"By mean final false-small-vehicle ratio (FSV), the lowest-FSV model is `{best_model['model_name']}` at `{_fmt(_float(best_model, 'mean_final_fsv'))}`. The highest-FSV model is `{worst_model['model_name']}` at `{_fmt(_float(worst_model, 'mean_final_fsv'))}`.",
        "",
        "The report is based on completed full S2 final-test, 12-angle closed-set inference. Open-vocabulary, full causal intervention, full context counterfactual, and full DeHub safety remain smoke-only or future full-benchmark work and are not promoted to full conclusions here.",
        "",
        "## Completion Status",
        "",
        _table(
            ["Model", "Status", "Expected", "Raw", "Canonical", "Failed"],
            [
                [
                    row["model_name"],
                    row["status"],
                    row["expected_tile_angle"],
                    row["raw_completed_tile_angle"],
                    row["canonical_completed_tile_angle"],
                    row["failed_tile_angle"],
                ]
                for row in status_rows
            ],
        ),
        "",
        "## False-Small-Vehicle Summary",
        "",
        "FSV is the ratio of predicted small-vehicle detections that do not match a small-vehicle GT at IoU 0.30. BG-FSV is the subset whose best overlap is background/no object. FR-SV is the fraction of final detections labeled small-vehicle.",
        "",
        _model_false_hub_table(false_hub_rows),
        "",
        "## Stage Decomposition Summary",
        "",
        _stage_table(stage_rows),
        "",
        "## Rotation Sensitivity",
        "",
        "RG-SV is the worst-angle FSV increase over angle 0 for each tile. ARG-SV is the average non-zero-angle increase over angle 0.",
        "",
        rotation_table,
        "",
        "## Error Concentration",
        "",
        _table(
            ["Top 1% Contribution", "Top 5% Contribution", "Gini", "HHI"],
            [
                [
                    _fmt(_float(concentration, "top_1pct_contribution")),
                    _fmt(_float(concentration, "top_5pct_contribution")),
                    _fmt(_float(concentration, "gini")),
                    _fmt(_float(concentration, "hhi"), 6),
                ]
            ],
        ),
        "",
        "## Generated Artifacts",
        "",
        "- `metrics/run_status_by_model.csv`: model completion status.",
        "- `metrics/full_closedset_false_hub_summary_by_model.csv`: model-level false-hub metrics.",
        "- `metrics/full_closedset_false_hub_summary_by_angle.csv`: model-angle false-hub metrics.",
        "- `metrics/full_closedset_rotation_gain.csv`: tile-level rotation sensitivity.",
        "- `metrics/full_closedset_concentration.csv`: concentration of false-SV events.",
        "- `metrics/stage_decomposition_summary.csv`: model-level stage availability and post-NMS FR-SV.",
        "- `metrics/stage_decomposition_tile_angle.csv`: tile-angle stage rows.",
        "",
        "## Scope Boundaries",
        "",
        "- This is a full closed-set report only.",
        "- Full open-vocabulary benchmark has not been run.",
        "- Full causal intervention has not been run; only smoke-scale classifier-channel intervention exists.",
        "- Full context counterfactual has not been run; only smoke-scale object/context image edits exist.",
        "- Full DeHub safety has not been run; only smoke-scale baseline-vs-repair comparison exists.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/05_eval_false_hub_taxonomy.py --run-dir experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle --iou-thr 0.3 --workers 16",
        "PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/06_stage_decomposition.py --run-dir experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle --workers 16",
        "PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/19_build_full_closedset_summary.py",
        "```",
        "",
    ]
    output.write_text("\n".join(sections))
    print(f"report={output}")


if __name__ == "__main__":
    main()

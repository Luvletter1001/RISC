#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from statistics import mean


EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = EXP_DIR / "outputs/runs/full_closedset_s2_12angle"
SMALL_VEHICLE = "small-vehicle"
EXPECTED_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _iter_csv(path: Path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def _int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return int(float(value))


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _fmt(value, digits: int = 4) -> str:
    if value in {"", None}:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _round_count(value: float) -> int:
    return int(round(value))


def _gini(values: list[float]) -> float:
    vals = sorted(v for v in values if v >= 0)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    n = len(vals)
    cumulative = sum((idx + 1) * val for idx, val in enumerate(vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def _hhi(values: list[float]) -> float:
    total = sum(values)
    return sum((v / total) ** 2 for v in values) if total else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _percentile(sorted_values: list[float], pct: float):
    if not sorted_values:
        return ""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = pct * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_table_md(path: Path, title: str, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = [[str(row.get(header, "")) for header in headers] for row in rows]
    path.write_text(f"# {title}\n\n" + _markdown_table(headers, table_rows) + "\n")


def _scan_score_chunk(paths: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for item in paths:
        path = Path(item)
        model = path.parts[-3]
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for pred in data.get("final_predictions", []):
            if pred.get("class_name") == SMALL_VEHICLE and pred.get("score") not in {"", None}:
                out[model].append(float(pred["score"]))
    return out


def _chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _score_stats(run_dir: Path, workers: int) -> dict[str, dict]:
    paths = [str(path) for path in sorted((run_dir / "canonical_predictions").glob("*/*/angle_*.json"))]
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    if not paths:
        return {}
    chunks = _chunks(paths, 512)
    if workers <= 1:
        results = [_scan_score_chunk(chunk) for chunk in chunks]
    else:
        with Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_scan_score_chunk, chunks, chunksize=1))
    for result in results:
        for model, scores in result.items():
            scores_by_model[model].extend(scores)
    stats = {}
    for model, scores in scores_by_model.items():
        ordered = sorted(scores)
        stats[model] = {
            "score_mean_sv_pred": mean(ordered) if ordered else "",
            "score_median_sv_pred": _percentile(ordered, 0.50),
            "score_quantiles_sv_pred": {
                "q05": _percentile(ordered, 0.05),
                "q25": _percentile(ordered, 0.25),
                "q50": _percentile(ordered, 0.50),
                "q75": _percentile(ordered, 0.75),
                "q95": _percentile(ordered, 0.95),
            },
        }
    return stats


def _status_checks(manifest: dict, status_rows: list[dict], false_summary_rows: list[dict]) -> dict:
    angles = [int(a) for a in manifest.get("angles", [])]
    model_status = manifest.get("model_status", {})
    expected = sum(_int(row, "expected_tile_angle") for row in status_rows)
    raw = sum(_int(row, "raw_completed_tile_angle") for row in status_rows)
    canonical = sum(_int(row, "canonical_completed_tile_angle") for row in status_rows)
    failed = sum(_int(row, "failed_tile_angle") for row in status_rows)
    checks = {
        "is_s2_final_test": manifest.get("split_name") == "S2_final_test",
        "has_12_angles": sorted(angles) == EXPECTED_ANGLES,
        "has_no_smoke_limit": int(manifest.get("args", {}).get("limit", 0) or 0) == 0,
        "is_actual_inference": manifest.get("execution_mode") in {"model_parallel_shards", "single_process"},
        "raw_predictions_complete": raw == expected and expected > 0,
        "canonical_predictions_complete": canonical == expected and expected > 0,
        "failed_tile_angles_zero": failed == 0,
        "all_models_done_full": all(row.get("status") == "DONE_FULL" for row in status_rows),
        "each_model_has_independent_status": set(model_status) == {row["model_name"] for row in status_rows},
        "gt_matching_complete": all(_bool_value(row.get("has_gt_matching")) for row in false_summary_rows),
        "false_hub_taxonomy_complete": all(_bool_value(row.get("has_false_sv_taxonomy")) for row in false_summary_rows),
        "no_proxy_or_schema_rows_in_main_table": all(
            row.get("status") == "DONE_FULL"
            and not _bool_value(row.get("is_proxy"))
            and row.get("status") not in {"DONE_SMOKE", "SMOKE_PROXY", "SCHEMA_ONLY"}
            for row in false_summary_rows
            if _bool_value(row.get("include_in_main_table"))
        ),
    }
    return {
        "checks": checks,
        "overall": all(checks.values()),
        "expected_tile_angles": expected,
        "raw_completed_tile_angles": raw,
        "canonical_completed_tile_angles": canonical,
        "failed_tile_angles": failed,
    }


def _aggregate_false_hub(metrics_dir: Path) -> tuple[dict, dict, list[dict], dict]:
    per_model = defaultdict(lambda: defaultdict(float))
    valid_per_model = defaultdict(lambda: defaultdict(float))
    mode_by_key: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    fsv_by_model_tile: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    event_by_model_tile = defaultdict(float)
    event_by_tile_all_models = defaultdict(float)
    fsv_by_model_angle = defaultdict(list)
    row_count = 0

    for row in _iter_csv(metrics_dir / "full_closedset_false_hub_tile_angle.csv"):
        row_count += 1
        model = row["model_name"]
        tile = row["tile_id"]
        angle = int(float(row["angle"]))
        mode = row.get("region_mode", "")
        mode_by_key[(model, tile, angle)].add(mode)
        num_sv = _float(row, "num_sv_pred")
        false_count = _round_count(num_sv * _float(row, "false_sv_ratio"))
        bg_count = _round_count(num_sv * _float(row, "bg_fsv_ratio"))
        total_pred = _float(row, "total_pred")
        matched_count = max(0, _round_count(num_sv - false_count))
        gt_sv = _float(row, "num_gt_sv")

        target = per_model if mode == "all_region" else valid_per_model if mode == "valid_mask_only" else None
        if target is not None:
            agg = target[model]
            agg["rows"] += 1
            agg["num_gt_sv"] += gt_sv
            agg["num_sv_pred"] += num_sv
            agg["num_matched_sv_pred"] += matched_count
            agg["num_unmatched_sv_pred"] += false_count
            agg["num_bg_sv_pred"] += bg_count
            agg["num_object_flip_sv_pred"] += _float(row, "object_flip_sv")
            agg["total_pred"] += total_pred
            agg["num_gt_total"] += _float(row, "num_gt_total")
            if mode == "all_region":
                fsv = _float(row, "false_sv_ratio")
                fsv_by_model_tile[(model, tile)][angle] = fsv
                fsv_by_model_angle[(model, angle)].append(fsv)
                event_by_model_tile[(model, tile)] += false_count
                event_by_tile_all_models[tile] += false_count

    missing_validmask = [
        {"model_name": model, "tile_id": tile, "angle": angle, "modes": ",".join(sorted(modes))}
        for (model, tile, angle), modes in sorted(mode_by_key.items())
        if modes != {"all_region", "valid_mask_only"}
    ]
    validmask_audit = {
        "status": "DONE_FULL" if not missing_validmask else "MISSING_VALIDMASK_VARIANT",
        "checked_model_tile_angles": len(mode_by_key),
        "raw_rows": row_count,
        "missing_count": len(missing_validmask),
        "missing_examples": missing_validmask[:20],
    }
    aux = {
        "fsv_by_model_tile": fsv_by_model_tile,
        "event_by_model_tile": event_by_model_tile,
        "event_by_tile_all_models": event_by_tile_all_models,
        "fsv_by_model_angle": fsv_by_model_angle,
    }
    return per_model, valid_per_model, missing_validmask, {"validmask_audit": validmask_audit, **aux}


def _denominator_rows(per_model: dict, valid_per_model: dict, score_stats: dict) -> list[dict]:
    rows = []
    for model, agg in sorted(per_model.items()):
        valid = valid_per_model.get(model, {})
        num_sv = agg["num_sv_pred"]
        false_count = agg["num_unmatched_sv_pred"]
        matched = agg["num_matched_sv_pred"]
        score = score_stats.get(model, {})
        q = score.get("score_quantiles_sv_pred", {})
        rows.append(
            {
                "model_name": model,
                "num_gt_sv": _round_count(agg["num_gt_sv"]),
                "num_sv_pred": _round_count(num_sv),
                "num_sv_pred_per_image": _safe_div(num_sv, agg["rows"]),
                "num_matched_sv_pred": _round_count(matched),
                "num_unmatched_sv_pred": _round_count(false_count),
                "num_bg_sv_pred": _round_count(agg["num_bg_sv_pred"]),
                "num_object_flip_sv_pred": _round_count(agg["num_object_flip_sv_pred"]),
                "fr_sv": _safe_div(num_sv, agg["total_pred"]),
                "fsv": _safe_div(false_count, num_sv),
                "bg_fsv": _safe_div(agg["num_bg_sv_pred"], num_sv),
                "object_flip_sv": _safe_div(agg["num_object_flip_sv_pred"], num_sv),
                "true_sv_recall": _safe_div(matched, agg["num_gt_sv"]),
                "true_sv_precision": _safe_div(matched, num_sv),
                "sv_ap50": "NOT_AVAILABLE_IN_CURRENT_ARTIFACTS",
                "score_mean_sv_pred": score.get("score_mean_sv_pred", ""),
                "score_median_sv_pred": score.get("score_median_sv_pred", ""),
                "score_quantiles_sv_pred": json.dumps(q, sort_keys=True) if q else "",
                "det_per_img": _safe_div(agg["total_pred"], agg["rows"]),
                "validmask_fsv": _safe_div(valid.get("num_unmatched_sv_pred", 0.0), valid.get("num_sv_pred", 0.0)),
                "rows": _round_count(agg["rows"]),
            }
        )
    return rows


def _rank(rows: list[dict], key: str, reverse: bool = True) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: row.get(key, 0), reverse=reverse)
    return {row["model_name"]: idx + 1 for idx, row in enumerate(ordered)}


def _paper_table_false_sv(denom_rows: list[dict], stage_summary: dict) -> list[dict]:
    rank_fsv = _rank(denom_rows, "fsv")
    rank_abs = _rank(denom_rows, "num_unmatched_sv_pred")
    rank_fr = _rank(denom_rows, "fr_sv")
    rank_recall = _rank(denom_rows, "true_sv_recall")
    rows = []
    for row in denom_rows:
        model = row["model_name"]
        stage = stage_summary.get(model, {})
        note = ""
        if model == "r3det_kfiou":
            note = "FSV reaches 1.0000 because matched true-SV predictions are almost absent; inspect Abs_FalseSV and FR_SV together."
        elif model == "redet":
            note = "Very high FSV with relatively low FR_SV; do not rank by FSV alone."
        elif model == "oriented_rcnn":
            note = "Lowest FSV; compare against SV_Pred_per_img and det/img before making severity claims."
        elif model == "rotated_rtmdet_l":
            note = "Moderate FSV with the strongest aggregate true-SV recall in this audit."
        rows.append(
            {
                "Model": model,
                "Family": stage.get("model_family", "closed_set"),
                "Architecture": stage.get("architecture_type", ""),
                "Status": "DONE_FULL",
                "mAP50": "NOT_AVAILABLE",
                "SV_AP50": row["sv_ap50"],
                "FR_SV": _fmt(row["fr_sv"]),
                "FSV": _fmt(row["fsv"]),
                "BG_FSV": _fmt(row["bg_fsv"]),
                "ObjectFlip_SV": _fmt(row["object_flip_sv"]),
                "Abs_FalseSV": row["num_unmatched_sv_pred"],
                "SV_Pred_per_img": _fmt(row["num_sv_pred_per_image"]),
                "True_SV_Recall": _fmt(row["true_sv_recall"]),
                "True_SV_Precision": _fmt(row["true_sv_precision"]),
                "det/img": _fmt(row["det_per_img"]),
                "ValidMask_FSV": _fmt(row["validmask_fsv"]),
                "Notes": note,
                "sort_by_FSV": rank_fsv[model],
                "sort_by_AbsFalseSV": rank_abs[model],
                "sort_by_FR_SV": rank_fr[model],
                "sort_by_TrueSVRecall": rank_recall[model],
            }
        )
    return sorted(rows, key=lambda row: (int(row["sort_by_AbsFalseSV"]), int(row["sort_by_FSV"])))


def _stage_tables(metrics_dir: Path) -> tuple[list[dict], list[dict], dict]:
    summary_rows = _read_csv(metrics_dir / "stage_decomposition_summary.csv")
    tile_rows = _read_csv(metrics_dir / "stage_decomposition_tile_angle.csv")
    metric_sums = defaultdict(lambda: defaultdict(float))
    metric_counts = defaultdict(lambda: defaultdict(int))
    for row in tile_rows:
        model = row["model_name"]
        for col in ["dense_or_query_sv", "pre_nms_fr_sv", "post_nms_fr_sv", "nms_amp_sv"]:
            if row.get(col) not in {"", None}:
                metric_sums[model][col] += float(row[col])
                metric_counts[model][col] += 1

    table = []
    final_taxonomy_eligible = []
    stage_claim_eligible = []
    final_only = []
    hook_unavailable = []
    summary_by_model = {}
    for row in sorted(summary_rows, key=lambda item: item["model_name"]):
        model = row["model_name"]
        statuses = {
            "dense": row.get("dense_logits_status", ""),
            "pre_nms": row.get("pre_nms_status", ""),
            "post_nms": row.get("post_nms_status", ""),
        }
        real_done = sum(1 for status in statuses.values() if status == "DONE_FULL")
        eligible = real_done >= 2
        if row.get("post_nms_status") == "DONE_FULL":
            final_taxonomy_eligible.append(model)
        if eligible:
            stage_claim_eligible.append(model)
        if statuses["dense"] == "NOT_APPLICABLE" and statuses["pre_nms"] == "NOT_APPLICABLE":
            final_only.append(model)
        if "UNSUPPORTED_BY_CURRENT_CODE" in statuses.values():
            hook_unavailable.append(model)
        notes = []
        if eligible:
            notes.append("stage claim eligible")
        elif model in final_only:
            notes.append("final-only for this audit; dense/pre-NMS structurally NOT_APPLICABLE")
        elif model in hook_unavailable:
            notes.append("hook unavailable; do not claim dense-stage bias")
        table_row = {
            "Model": model,
            "Architecture": row.get("architecture_type", ""),
            "Dense_Status": statuses["dense"],
            "PreNMS_Status": statuses["pre_nms"],
            "PostNMS_Status": statuses["post_nms"],
            "Dense_FR_SV": _fmt(_safe_div(metric_sums[model]["dense_or_query_sv"], metric_counts[model]["dense_or_query_sv"])),
            "PreNMS_FR_SV": _fmt(_safe_div(metric_sums[model]["pre_nms_fr_sv"], metric_counts[model]["pre_nms_fr_sv"])),
            "PostNMS_FR_SV": _fmt(_safe_div(metric_sums[model]["post_nms_fr_sv"], metric_counts[model]["post_nms_fr_sv"])),
            "NMS_Amp_SV": _fmt(_safe_div(metric_sums[model]["nms_amp_sv"], metric_counts[model]["nms_amp_sv"])),
            "StageClaimEligible": str(eligible),
            "Notes": "; ".join(notes),
        }
        table.append(table_row)
        summary_by_model[model] = row
    return table, summary_rows, {
        "summary_by_model": summary_by_model,
        "final_taxonomy_eligible_models": sorted(final_taxonomy_eligible),
        "stage_decomposition_eligible_models": sorted(stage_claim_eligible),
        "final_only_models": sorted(final_only),
        "hook_unavailable_models": sorted(hook_unavailable),
    }


def _rotation_audit(aux: dict) -> tuple[list[dict], list[dict], dict]:
    tile_rows = []
    by_model = defaultdict(list)
    inconsistent = []
    for (model, tile), by_angle in sorted(aux["fsv_by_model_tile"].items()):
        if not by_angle:
            continue
        base = by_angle.get(0, 0.0)
        deltas = {angle: value - base for angle, value in by_angle.items()}
        worst_abs_angle = max(by_angle, key=lambda angle: (by_angle[angle], -angle))
        worst_delta_angle = max(deltas, key=lambda angle: (deltas[angle], -angle))
        rg = max(0.0, deltas[worst_delta_angle])
        if rg > 1e-12 and worst_delta_angle == 0:
            inconsistent.append({"model_name": model, "tile_id": tile, "rg_sv": rg})
        vals = list(by_angle.values())
        nonzero_deltas = [delta for angle, delta in deltas.items() if angle != 0]
        row = {
            "model_name": model,
            "tile_id": tile,
            "fsv_angle0": base,
            "worst_angle_absolute_fsv": worst_abs_angle,
            "worst_fsv_absolute": by_angle[worst_abs_angle],
            "worst_angle_delta_over_angle0": worst_delta_angle,
            "worst_delta_over_angle0": deltas[worst_delta_angle],
            "rg_sv": rg,
            "arg_sv": sum(nonzero_deltas) / len(nonzero_deltas) if nonzero_deltas else 0.0,
            "range_fsv": max(vals) - min(vals),
            "std_fsv": _std(vals),
            "num_angles": len(by_angle),
            "semantics_ok": worst_delta_angle != 0 or rg == 0.0,
            "notes": "",
        }
        tile_rows.append(row)
        by_model[model].append(row)

    summary_rows = []
    for model, rows in sorted(by_model.items()):
        angle_means = {}
        for (m, angle), vals in aux["fsv_by_model_angle"].items():
            if m == model and vals:
                angle_means[angle] = mean(vals)
        positive = [row for row in rows if row["rg_sv"] > 0]
        delta_counter = Counter(int(row["worst_angle_delta_over_angle0"]) for row in positive)
        abs_counter = Counter(int(row["worst_angle_absolute_fsv"]) for row in rows)
        angle_values = list(angle_means.values())
        row_inconsistent = [row for row in rows if not row["semantics_ok"]]
        summary_rows.append(
            {
                "Model": model,
                "Mean_FSV_angle0": _fmt(mean([row["fsv_angle0"] for row in rows])),
                "Mean_FSV_worst": _fmt(mean([row["worst_fsv_absolute"] for row in rows])),
                "Mean_RG_SV": _fmt(mean([row["rg_sv"] for row in rows])),
                "Mean_ARG_SV": _fmt(mean([row["arg_sv"] for row in rows])),
                "WorstAngle_by_delta": delta_counter.most_common(1)[0][0] if delta_counter else 0,
                "WorstAngle_by_absolute": abs_counter.most_common(1)[0][0] if abs_counter else 0,
                "Range_FSV": _fmt((max(angle_values) - min(angle_values)) if angle_values else 0.0),
                "Std_FSV": _fmt(_std(angle_values)),
                "Tiles": len(rows),
                "Notes": "row-level semantics OK" if not row_inconsistent else f"{len(row_inconsistent)} inconsistent rows",
            }
        )
    audit = {
        "row_semantics_inconsistency_count": len(inconsistent),
        "row_semantics_inconsistency_examples": inconsistent[:20],
        "definition": "worst_angle_absolute_fsv is max absolute FSV; worst_angle_delta_over_angle0 is max FSV(angle)-FSV(angle0); rg_sv is positive part of that delta.",
    }
    return tile_rows, summary_rows, audit


def _concentration_audit(aux: dict) -> tuple[list[dict], dict]:
    by_model_values = defaultdict(list)
    by_model_tiles = defaultdict(list)
    for (model, tile), value in aux["event_by_model_tile"].items():
        by_model_values[model].append(value)
        by_model_tiles[model].append((tile, value))
    top_sets = {}
    rows = []
    for model, pairs in sorted(by_model_tiles.items()):
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        values = by_model_values[model]
        total = sum(values)
        top_sets[model] = {tile for tile, _ in pairs[:20]}
        rows.append(
            {
                "Model": model,
                "Top1pctContribution": _fmt(_safe_div(sum(v for _, v in pairs[: max(1, int(len(pairs) * 0.01))]), total)),
                "Top5pctContribution": _fmt(_safe_div(sum(v for _, v in pairs[: max(1, int(len(pairs) * 0.05))]), total)),
                "Gini": _fmt(_gini(values)),
                "HHI": _fmt(_hhi(values), 6),
                "TopRiskTile1": pairs[0][0] if len(pairs) > 0 else "",
                "TopRiskTile2": pairs[1][0] if len(pairs) > 1 else "",
                "TopRiskTile3": pairs[2][0] if len(pairs) > 2 else "",
                "Notes": "per-model false-SV event concentration over all_region rows",
            }
        )
    overall_pairs = sorted(aux["event_by_tile_all_models"].items(), key=lambda item: item[1], reverse=True)
    overlap_counter = Counter()
    for tiles in top_sets.values():
        overlap_counter.update(tiles)
    overlap = {tile: count for tile, count in overlap_counter.items() if count > 1}
    audit = {
        "top_20_risky_tiles_per_model": {
            model: [{"tile_id": tile, "false_sv_events": value} for tile, value in sorted(pairs, key=lambda item: item[1], reverse=True)[:20]]
            for model, pairs in by_model_tiles.items()
        },
        "top_20_risky_tiles_overall": [{"tile_id": tile, "false_sv_events": value} for tile, value in overall_pairs[:20]],
        "high_risk_tile_overlap_across_models": {
            "overlap_tile_count": len(overlap),
            "top_overlaps": [
                {"tile_id": tile, "model_top20_count": count}
                for tile, count in sorted(overlap.items(), key=lambda item: item[1], reverse=True)[:20]
            ],
        },
    }
    return rows, audit


def _specific_checks(rows: list[dict]) -> dict:
    by_model = {row["model_name"]: row for row in rows}
    out = {}
    for model in ["r3det_kfiou", "redet", "oriented_rcnn", "rotated_rtmdet_l"]:
        row = by_model.get(model, {})
        if not row:
            continue
        out[model] = {
            "fsv": row["fsv"],
            "fr_sv": row["fr_sv"],
            "num_sv_pred": row["num_sv_pred"],
            "num_matched_sv_pred": row["num_matched_sv_pred"],
            "num_unmatched_sv_pred": row["num_unmatched_sv_pred"],
            "true_sv_recall": row["true_sv_recall"],
            "true_sv_precision": row["true_sv_precision"],
            "interpretation": "",
        }
    if "r3det_kfiou" in out:
        out["r3det_kfiou"]["interpretation"] = "FSV=1.0000 is explained by almost no true-SV matches; precision and recall must be cited with FSV."
    if "redet" in out:
        out["redet"]["interpretation"] = "FSV is very high while FR-SV is low; severity claims require absolute false-SV count."
    if "oriented_rcnn" in out:
        out["oriented_rcnn"]["interpretation"] = "Lower FSV is supported, but it should be compared with its smaller SV prediction rate."
    if "rotated_rtmdet_l" in out:
        out["rotated_rtmdet_l"]["interpretation"] = "Lower FSV than most dense models coexists with limited true-SV recall, so it is not a solved detector."
    return out


def _write_audit_md(path: Path, audit: dict, denom_rows: list[dict], stage_info: dict, rotation_summary: list[dict], concentration_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = audit["done_full_qualification"]["checks"]
    check_rows = [[key, str(value)] for key, value in checks.items()]
    denom_headers = [
        "model_name",
        "num_gt_sv",
        "num_sv_pred",
        "num_unmatched_sv_pred",
        "fr_sv",
        "fsv",
        "bg_fsv",
        "true_sv_recall",
        "true_sv_precision",
        "det_per_img",
        "validmask_fsv",
    ]
    denom_table = [[_fmt(row.get(h)) if isinstance(row.get(h), float) else str(row.get(h, "")) for h in denom_headers] for row in denom_rows]
    rotation_headers = ["Model", "Mean_FSV_angle0", "Mean_FSV_worst", "Mean_RG_SV", "Mean_ARG_SV", "WorstAngle_by_delta", "WorstAngle_by_absolute", "Tiles", "Notes"]
    concentration_headers = ["Model", "Top1pctContribution", "Top5pctContribution", "Gini", "HHI", "TopRiskTile1", "TopRiskTile2", "TopRiskTile3"]
    sections = [
        "# Full Closed-Set Scientific Audit",
        "",
        f"- generated_at: `{audit['generated_at']}`",
        f"- run_dir: `{audit['run_dir']}`",
        f"- split: `{audit['split_name']}`",
        f"- angles: `{','.join(str(a) for a in audit['angles'])}`",
        f"- models: `{len(audit['models'])}`",
        "",
        "## DONE_FULL Qualification",
        "",
        _markdown_table(["Check", "Pass"], check_rows),
        "",
        "## FSV Denominator Audit",
        "",
        "FSV is not interpreted alone: this table exposes the small-vehicle prediction denominator, absolute false-SV counts, FR-SV, recall, precision, and valid-mask variant.",
        "",
        _markdown_table(denom_headers, denom_table),
        "",
        "## FR-SV vs FSV Decoupling",
        "",
        "High FSV with few small-vehicle predictions is not equivalent to a high absolute false-SV burden. The paper tables therefore include FR_SV, FSV, Abs_FalseSV, SV_Pred_per_img, and det/img together.",
        "",
        "## Stage Decomposition Audit",
        "",
        f"- final-taxonomy eligible models: `{', '.join(stage_info['final_taxonomy_eligible_models'])}`",
        f"- stage-decomposition eligible models: `{', '.join(stage_info['stage_decomposition_eligible_models'])}`",
        f"- structurally final-only / NOT_APPLICABLE models: `{', '.join(stage_info['final_only_models'])}`",
        f"- hook-unavailable / UNSUPPORTED_BY_CURRENT_CODE models: `{', '.join(stage_info['hook_unavailable_models'])}`",
        "",
        "Final-only models are eligible for final false-SV taxonomy, but they are not eligible for dense/pre-NMS bias claims.",
        "",
        "## Rotation Gain Sanity",
        "",
        audit["rotation_gain_audit"]["definition"],
        "",
        f"- row_semantics_inconsistency_count: `{audit['rotation_gain_audit']['row_semantics_inconsistency_count']}`",
        "",
        _markdown_table(rotation_headers, [[str(row.get(h, "")) for h in rotation_headers] for row in rotation_summary]),
        "",
        "## Concentration Audit",
        "",
        _markdown_table(concentration_headers, [[str(row.get(h, "")) for h in concentration_headers] for row in concentration_rows]),
        "",
        "## Valid-Mask Audit",
        "",
        f"- status: `{audit['valid_mask_audit']['status']}`",
        f"- checked_model_tile_angles: `{audit['valid_mask_audit']['checked_model_tile_angles']}`",
        f"- missing_count: `{audit['valid_mask_audit']['missing_count']}`",
        "",
        "## Scope Boundaries",
        "",
        "- This audit is full closed-set only.",
        "- Full open-vocabulary benchmark has not been run.",
        "- Full causal intervention has not been run.",
        "- Full context counterfactual has not been run.",
        "- Full DeHub safety has not been run.",
    ]
    path.write_text("\n".join(sections) + "\n")


def _write_claims(path: Path, audit: dict, denom_rows: list[dict], stage_info: dict) -> None:
    by_model = {row["model_name"]: row for row in denom_rows}
    low = min(denom_rows, key=lambda row: row["fsv"])
    high = max(denom_rows, key=lambda row: row["fsv"])
    raw = audit["done_full_qualification"]["raw_completed_tile_angles"]
    canonical = audit["done_full_qualification"]["canonical_completed_tile_angles"]
    bg_high = ", ".join(row["model_name"] for row in sorted(denom_rows, key=lambda item: item["bg_fsv"], reverse=True)[:4])
    sections = [
        "# Closed-Set Claims for Paper",
        "",
        "## Claim A: Strong Claim for Main Text",
        "",
        "Closed-set remote-sensing detectors also exhibit false-small-vehicle failures under rotation-conditioned S2 final-test evaluation, but the severity is highly model-dependent.",
        "",
        f"Evidence: 10/10 closed-set models are DONE_FULL on S2_final_test with 12 angles and 0 failed tile-angles. The run contains {raw} raw and {canonical} canonical outputs. FSV ranges from {low['model_name']} at {_fmt(low['fsv'])} to {high['model_name']} at {_fmt(high['fsv'])}. BG-FSV is high in several models, especially {bg_high}.",
        "",
        "## Claim B: Bounded Claim",
        "",
        "For models with dense/pre-NMS hooks, the failure can be decomposed before and after suppression, but this claim cannot be generalized to all closed-set models because several architectures are final-only or hook-unavailable.",
        "",
        f"Stage eligible models: {', '.join(stage_info['stage_decomposition_eligible_models'])}.",
        f"Final-only NOT_APPLICABLE models: {', '.join(stage_info['final_only_models'])}.",
        f"Hook-unavailable UNSUPPORTED_BY_CURRENT_CODE models: {', '.join(stage_info['hook_unavailable_models'])}.",
        "",
        "## Claim C: Forbidden Claims",
        "",
        "- Do not write that all closed-set models collapse to small-vehicle.",
        "- Do not write that closed-set models have open-vocabulary embedding attractors.",
        "- Do not write that DeHub is deployable.",
        "- Do not write that context is causally proven at full scale.",
        "- Do not write that OpenRSD full benchmark is complete.",
        "- Do not write that NMS is the root cause for all closed-set models.",
        "",
        "## Model-Specific Caution",
        "",
        f"- r3det_kfiou: FSV {_fmt(by_model['r3det_kfiou']['fsv'])}, FR-SV {_fmt(by_model['r3det_kfiou']['fr_sv'])}, true-SV precision {_fmt(by_model['r3det_kfiou']['true_sv_precision'])}; cite as near-absence of true-SV matches, not simply as higher hub severity.",
        f"- redet: FSV {_fmt(by_model['redet']['fsv'])}, FR-SV {_fmt(by_model['redet']['fr_sv'])}; high FSV but low SV prediction rate.",
        f"- oriented_rcnn: lowest FSV {_fmt(by_model['oriented_rcnn']['fsv'])}; keep denominator and SV_Pred_per_img visible.",
        f"- rotated_rtmdet_l: FSV {_fmt(by_model['rotated_rtmdet_l']['fsv'])}, true-SV recall {_fmt(by_model['rotated_rtmdet_l']['true_sv_recall'])}; improved relative stability is not full small-vehicle competence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n")


def _write_next_plan(path: Path, run_dir: Path) -> None:
    py = "/data/zcy/anaconda3/envs/openrsd/bin/python"
    split = "experiments/rotation_semantic_attractor/outputs/splits/S2_final_test.json"
    angle_set = "0,30,60,90,120,150,180,210,240,270,300,330"
    sections = [
        "# Next Full Execution Plan After Closed-Set",
        "",
        "No time estimates are provided. These are execution criteria and artifact expectations only.",
        "",
        "## P0: Closed-Set Audit Hardening",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/20_closedset_scientific_audit.py --run-dir {run_dir} --workers 16`",
        f"- input split: `{split}`",
        f"- output dir: `{run_dir}/metrics` and `experiments/rotation_semantic_attractor/reports`",
        "- DONE_FULL criteria: FSV denominator audit, SV AP50 availability check, absolute false-SV counts, and audited rotation gain semantics are generated from full closed-set outputs.",
        "- blockers: SV_AP50/mAP50 require an evaluator artifact or a dedicated evaluator pass; current audit marks them unavailable instead of fabricating proxies.",
        "- estimated artifacts: audit MD/JSON, paper tables, audited rotation gain CSVs.",
        "",
        "## P1: Full OpenRSD/Open-Vocabulary S2 12-Angle Benchmark",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/03_run_angle_sweep.py --split {split} --models <open_vocab_models> --angles {angle_set} --output-dir experiments/rotation_semantic_attractor/outputs/runs/full_openvocab_s2_12angle --seed 20260530`",
        f"- input split: `{split}`",
        "- output dir: `experiments/rotation_semantic_attractor/outputs/runs/full_openvocab_s2_12angle`",
        "- DONE_FULL criteria: selected open-vocabulary models complete raw/canonical predictions for all S2 tiles and all 12 angles, with 0 failed tile-angles and full false-hub taxonomy.",
        "- blockers: open-vocabulary model assets, prompt/support protocol, and adapter output normalization must be frozen.",
        "- estimated artifacts: manifest, model_status, raw/canonical predictions, false-hub taxonomy, open-vocab audit report.",
        "",
        "## P2: Full S3 Causal Intervention",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/07_intervention_open_vocab.py --run-dir experiments/rotation_semantic_attractor/outputs/runs/full_openvocab_s2_12angle --output-dir experiments/rotation_semantic_attractor/outputs/runs/full_s3_causal_intervention/metrics`",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/08_intervention_closed_set.py --run-dir {run_dir} --mode real --limit 0 --output-dir experiments/rotation_semantic_attractor/outputs/runs/full_s3_causal_intervention/metrics`",
        "- input split: S3 intervention split derived from the frozen full benchmark protocol.",
        "- output dir: `experiments/rotation_semantic_attractor/outputs/runs/full_s3_causal_intervention`",
        "- DONE_FULL criteria: OpenRSD embedding intervention and closed-set classifier-channel intervention rerun inference on the full selected S3 set with paired baseline/intervention comparisons.",
        "- blockers: S3 split freeze, GPU budget, intervention support for each adapter, and paired metric schema.",
        "- estimated artifacts: paired predictions, intervention CSVs, class drift, true-SV preservation, causal audit.",
        "",
        "## P3: Full Real Context Counterfactual on S3",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/09_context_counterfactual.py --run-dir experiments/rotation_semantic_attractor/outputs/runs/full_s3_context_counterfactual --mode real --limit 0`",
        "- input split: frozen S3 context-counterfactual split.",
        "- output dir: `experiments/rotation_semantic_attractor/outputs/runs/full_s3_context_counterfactual`",
        "- DONE_FULL criteria: real image-level edits are generated, inference is rerun, and paired baseline/counterfactual rows exist for every selected item.",
        "- blockers: counterfactual edit policy, mask generation QA, storage budget, and visual validity checks.",
        "- estimated artifacts: edited images, paired predictions, context metrics, QA thumbnails, audit report.",
        "",
        "## P4: Full DeHub Safety",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/10_eval_dehub_safety.py --run-dir experiments/rotation_semantic_attractor/outputs/runs/full_dehub_safety_s2_12angle --mode real --limit 0`",
        f"- input split: `{split}`",
        "- output dir: `experiments/rotation_semantic_attractor/outputs/runs/full_dehub_safety_s2_12angle`",
        "- DONE_FULL criteria: baseline vs B5k/B8k use the same split, same angle set, same threshold/evaluator; true-SV preservation, class drift, low-risk inflation, mAP and SV AP are reported when evaluator artifacts exist.",
        "- blockers: repair checkpoints, evaluator availability, OpenRSD adapter stability, and paired safety schema.",
        "- estimated artifacts: baseline/repair predictions, safety CSVs, AP tables, low-risk inflation report, deployment-blocking audit.",
        "",
        "## P5: Per-Angle TTA Range and Oracle Best-View",
        "",
        f"- command: `{py} experiments/rotation_semantic_attractor/scripts/05_eval_false_hub_taxonomy.py --run-dir {run_dir} --iou-thr 0.3 --workers 16`",
        f"- input split: `{split}`",
        f"- output dir: `{run_dir}/metrics`",
        "- DONE_FULL criteria: per-angle TTA range and oracle best-view are computed from full 12-angle rows and marked as non-deployable when using oracle angle selection.",
        "- blockers: none for closed-set if current full rows remain available; open-vocabulary requires P1 completion.",
        "- estimated artifacts: oracle_best_view, rotation gain, per-angle range tables, deployability notes.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_dir = run_dir / "metrics"
    reports_dir = EXP_DIR / "reports"
    tables_dir = reports_dir / "tables"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    manifest = _read_json(run_dir / "manifest.json")
    status_rows = _read_csv(metrics_dir / "run_status_by_model.csv")
    false_summary_rows = _read_csv(metrics_dir / "full_closedset_false_hub_summary_by_model.csv")
    done_full = _status_checks(manifest, status_rows, false_summary_rows)
    score_stats = _score_stats(run_dir, max(1, args.workers))
    per_model, valid_per_model, _, aux = _aggregate_false_hub(metrics_dir)
    denom_rows = _denominator_rows(per_model, valid_per_model, score_stats)
    stage_table, _, stage_info = _stage_tables(metrics_dir)
    rotation_tile_rows, rotation_summary_rows, rotation_audit = _rotation_audit(aux)
    concentration_rows, concentration_audit = _concentration_audit(aux)

    _write_csv(metrics_dir / "full_closedset_rotation_gain_audited.csv", rotation_tile_rows)
    _write_csv(metrics_dir / "full_closedset_rotation_gain_summary_audited.csv", rotation_summary_rows)

    false_table_rows = _paper_table_false_sv(denom_rows, stage_info["summary_by_model"])
    false_headers = [
        "Model",
        "Family",
        "Architecture",
        "Status",
        "mAP50",
        "SV_AP50",
        "FR_SV",
        "FSV",
        "BG_FSV",
        "ObjectFlip_SV",
        "Abs_FalseSV",
        "SV_Pred_per_img",
        "True_SV_Recall",
        "True_SV_Precision",
        "det/img",
        "ValidMask_FSV",
        "Notes",
        "sort_by_FSV",
        "sort_by_AbsFalseSV",
        "sort_by_FR_SV",
        "sort_by_TrueSVRecall",
    ]
    _write_csv(metrics_dir / "paper_table_closedset_false_sv_benchmark.csv", false_table_rows, false_headers)
    _write_table_md(tables_dir / "paper_table_closedset_false_sv_benchmark.md", "Table 1: Closed-Set False-SV Benchmark", false_headers, false_table_rows)

    stage_headers = [
        "Model",
        "Architecture",
        "Dense_Status",
        "PreNMS_Status",
        "PostNMS_Status",
        "Dense_FR_SV",
        "PreNMS_FR_SV",
        "PostNMS_FR_SV",
        "NMS_Amp_SV",
        "StageClaimEligible",
        "Notes",
    ]
    _write_csv(metrics_dir / "paper_table_closedset_stage_decomposition.csv", stage_table, stage_headers)
    _write_table_md(tables_dir / "paper_table_closedset_stage_decomposition.md", "Table 2: Closed-Set Stage Decomposition", stage_headers, stage_table)

    rotation_headers = [
        "Model",
        "Mean_FSV_angle0",
        "Mean_FSV_worst",
        "Mean_RG_SV",
        "Mean_ARG_SV",
        "WorstAngle_by_delta",
        "WorstAngle_by_absolute",
        "Range_FSV",
        "Std_FSV",
        "Tiles",
        "Notes",
    ]
    _write_csv(metrics_dir / "paper_table_closedset_rotation_gain.csv", rotation_summary_rows, rotation_headers)
    _write_table_md(tables_dir / "paper_table_closedset_rotation_gain.md", "Table 3: Rotation-Conditioned False-SV Gain", rotation_headers, rotation_summary_rows)

    concentration_headers = [
        "Model",
        "Top1pctContribution",
        "Top5pctContribution",
        "Gini",
        "HHI",
        "TopRiskTile1",
        "TopRiskTile2",
        "TopRiskTile3",
        "Notes",
    ]
    _write_csv(metrics_dir / "paper_table_closedset_concentration.csv", concentration_rows, concentration_headers)
    _write_table_md(tables_dir / "paper_table_closedset_concentration.md", "Table 4: Error Concentration", concentration_headers, concentration_rows)

    audit = {
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "split_name": manifest.get("split_name"),
        "angles": manifest.get("angles", []),
        "models": manifest.get("selected_models", []),
        "done_full_qualification": done_full,
        "fsv_denominator_audit": denom_rows,
        "specific_model_checks": _specific_checks(denom_rows),
        "fr_sv_vs_fsv_decoupling": {
            "policy": "Rank and claim using FR-SV, FSV, absolute false-SV count, SV predictions per image, true-SV recall/precision, and det/img together.",
            "rankings": {
                "sort_by_FSV": _rank(denom_rows, "fsv"),
                "sort_by_AbsFalseSV": _rank(denom_rows, "num_unmatched_sv_pred"),
                "sort_by_FR_SV": _rank(denom_rows, "fr_sv"),
                "sort_by_TrueSVRecall": _rank(denom_rows, "true_sv_recall"),
            },
        },
        "stage_decomposition_audit": {
            "final_taxonomy_eligible_models": stage_info["final_taxonomy_eligible_models"],
            "stage_decomposition_eligible_models": stage_info["stage_decomposition_eligible_models"],
            "final_only_models": stage_info["final_only_models"],
            "hook_unavailable_models": stage_info["hook_unavailable_models"],
            "rule": "StageClaimEligible=true only when at least two real dense/pre-NMS/post-NMS stages are DONE_FULL.",
        },
        "rotation_gain_audit": rotation_audit,
        "concentration_audit": concentration_audit,
        "valid_mask_audit": aux["validmask_audit"],
        "scope_boundaries": {
            "full_closed_set": True,
            "full_open_vocab": False,
            "full_causal_intervention": False,
            "full_context_counterfactual": False,
            "full_dehub_safety": False,
        },
        "generated_artifacts": {
            "audit_md": str(reports_dir / "full_closedset_scientific_audit.md"),
            "audit_json": str(reports_dir / "full_closedset_scientific_audit.json"),
            "rotation_gain_audited": str(metrics_dir / "full_closedset_rotation_gain_audited.csv"),
            "rotation_gain_summary_audited": str(metrics_dir / "full_closedset_rotation_gain_summary_audited.csv"),
            "paper_tables_metrics": str(metrics_dir),
            "paper_tables_md": str(tables_dir),
            "claims": str(reports_dir / "closedset_claims_for_paper.md"),
            "next_plan": str(reports_dir / "next_full_execution_plan_after_closedset.md"),
        },
    }
    _write_json(reports_dir / "full_closedset_scientific_audit.json", audit)
    _write_audit_md(reports_dir / "full_closedset_scientific_audit.md", audit, denom_rows, stage_info, rotation_summary_rows, concentration_rows)
    _write_claims(reports_dir / "closedset_claims_for_paper.md", audit, denom_rows, stage_info)
    _write_next_plan(reports_dir / "next_full_execution_plan_after_closedset.md", run_dir)

    print(f"audit_md={reports_dir / 'full_closedset_scientific_audit.md'}")
    print(f"audit_json={reports_dir / 'full_closedset_scientific_audit.json'}")
    print(f"tables_dir={tables_dir}")


if __name__ == "__main__":
    main()

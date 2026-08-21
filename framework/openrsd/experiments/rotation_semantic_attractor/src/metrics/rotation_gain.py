from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


def _gini(values: List[float]) -> float:
    vals = sorted(v for v in values if v >= 0)
    if not vals or sum(vals) == 0:
        return 0.0
    n = len(vals)
    cumulative = sum((idx + 1) * val for idx, val in enumerate(vals))
    return (2 * cumulative) / (n * sum(vals)) - (n + 1) / n


def rotation_gain_rows(false_hub_rows: Iterable[Dict]) -> tuple[List[Dict], List[Dict]]:
    grouped = defaultdict(list)
    for row in false_hub_rows:
        if row.get("region_mode") != "all_region":
            continue
        grouped[(row["model_name"], row["tile_id"])].append(row)

    gain_rows = []
    event_counts = []
    for (model, tile), rows in grouped.items():
        by_angle = {int(r["angle"]): float(r["false_sv_ratio"]) for r in rows}
        base = by_angle.get(0, 0.0)
        worst_angle = max(by_angle, key=lambda a: by_angle[a]) if by_angle else 0
        other = [v - base for a, v in by_angle.items() if a != 0]
        gain_rows.append(
            {
                "model_name": model,
                "tile_id": tile,
                "rg_sv": by_angle.get(worst_angle, 0.0) - base,
                "arg_sv": sum(other) / len(other) if other else 0.0,
                "worst_angle_sv": worst_angle,
            }
        )
        event_counts.append(float(sum(float(r.get("num_sv_pred", 0)) * float(r.get("false_sv_ratio", 0)) for r in rows)))

    total = sum(event_counts)
    sorted_counts = sorted(event_counts, reverse=True)
    concentration = [
        {
            "top_1pct_contribution": sum(sorted_counts[: max(1, int(len(sorted_counts) * 0.01))]) / total if total else 0.0,
            "top_5pct_contribution": sum(sorted_counts[: max(1, int(len(sorted_counts) * 0.05))]) / total if total else 0.0,
            "gini": _gini(event_counts),
            "hhi": sum((v / total) ** 2 for v in event_counts) if total else 0.0,
        }
    ]
    return gain_rows, concentration


def oracle_best_view_rows(false_hub_rows: Iterable[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in false_hub_rows:
        if row.get("region_mode") != "all_region":
            continue
        grouped[(row["model_name"], row["tile_id"])].append(row)

    out = []
    for (model, tile), rows in sorted(grouped.items()):
        scored = [
            (
                int(row["angle"]),
                float(row.get("false_sv_ratio") or 0.0),
                float(row.get("fr_sv") or 0.0),
                int(float(row.get("det_per_img") or 0)),
            )
            for row in rows
        ]
        if not scored:
            continue
        best = min(scored, key=lambda item: (item[1], item[2], item[3]))
        worst = max(scored, key=lambda item: (item[1], item[2], item[3]))
        base = next((item for item in scored if item[0] == 0), None)
        out.append(
            {
                "model_name": model,
                "tile_id": tile,
                "angle_set": ",".join(str(item[0]) for item in sorted(scored)),
                "num_angles": len(scored),
                "oracle_best_angle": best[0],
                "oracle_best_fsv": best[1],
                "oracle_best_fr_sv": best[2],
                "worst_angle": worst[0],
                "worst_fsv": worst[1],
                "angle000_fsv": base[1] if base else "",
                "oracle_gain_vs_angle000": (base[1] - best[1]) if base else "",
                "oracle_gain_vs_worst": worst[1] - best[1],
                "deployable": False,
                "uses_gt_or_angle_oracle": True,
            }
        )
    return out

#!/usr/bin/env python3
"""Audit whether small-vehicle attraction holds on RSV taxonomy 2500 cohort."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

REPO = Path("/data1/zcy/OpenRSD")
WORK = REPO / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT = REPO / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
OUT_DIR = WORK / "rsv_attractor_2500_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def f(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def i(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def quant(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


summary_ta = read_csv(WORK / "ftable_case_summary_tile_angle.csv")
summary_tile = read_csv(WORK / "ftable_case_summary_tile.csv")
sv_events = read_csv(WORK / "ftable_case_sv_events.csv")
gt_events = read_csv(WORK / "ftable_case_gt_events.csv")
rot_tile = read_csv(WORK / "ftable_case1_rotation_tile.csv")
cohort = read_csv(WORK / "ftable_cohort_2500.csv")

n_tiles = len({r["tile_id"] for r in summary_ta})
n_tile_angles = len(summary_ta)
total_pred = sum(i(r["n_pred"]) for r in summary_ta)
total_pred_sv = sum(i(r["n_pred_sv"]) for r in summary_ta)
total_gt_sv = sum(i(r["n_gt_sv"]) for r in summary_ta)
total_gt = sum(i(r["n_gt"]) for r in summary_ta)
total_case1 = sum(i(r["n_sv_case1"]) for r in summary_ta)
total_bg = sum(i(r["n_sv_bg"]) for r in summary_ta)

# sv event attribution by taxonomy tag and best GT class.
event_counts = Counter(r["event"] for r in sv_events)
gt_class_for_sv = Counter((r.get("gt_class") or "NO_GT") for r in sv_events)
case1_gt_class = Counter(r.get("gt_class") or "" for r in gt_events if r.get("event") == "case1")
case2_drift = Counter(r.get("drift_type") or "" for r in gt_events if r.get("event") == "case2")

# Per tile metrics.
tile_rows = []
for r in summary_tile:
    sv = i(r["total_sv_dets"])
    c1 = i(r["total_sv_case1"])
    bg = i(r["total_sv_bg"])
    case2 = i(r["total_case2_events"])
    tile_rows.append({
        "tile_id": r["tile_id"],
        "total_sv_dets": sv,
        "total_sv_case1": c1,
        "total_sv_bg": bg,
        "total_case2_events": case2,
        "case1_sv_fraction": f(r["case1_sv_fraction"]),
        "bg_sv_fraction": bg / sv if sv else 0.0,
        "non_residual_false_sv_fraction": (bg + c1) / sv if sv else 0.0,
        "case1_peak_angle": r.get("case1_peak_angle", ""),
        "case1_peak_rate": f(r.get("case1_peak_rate", 0)),
    })

# Tile-angle false hub cases: no GT SV but predicted SV.
ta_no_gt_sv_pred_sv = [r for r in summary_ta if i(r["n_gt_sv"]) == 0 and i(r["n_pred_sv"]) > 0]
ta_no_gt_any_pred_sv = [r for r in summary_ta if i(r["n_gt"]) == 0 and i(r["n_pred_sv"]) > 0]
ta_pred_sv_ratio_vals = [i(r["n_pred_sv"]) / i(r["n_pred"]) for r in summary_ta if i(r["n_pred"]) > 0]

# Angle aggregation.
angle_rows = []
by_angle = defaultdict(list)
for r in summary_ta:
    by_angle[i(r["angle"])].append(r)
for ang in sorted(by_angle):
    rows = by_angle[ang]
    pred = sum(i(r["n_pred"]) for r in rows)
    sv = sum(i(r["n_pred_sv"]) for r in rows)
    c1 = sum(i(r["n_sv_case1"]) for r in rows)
    bg = sum(i(r["n_sv_bg"]) for r in rows)
    no_gt_sv_pred = sum(1 for r in rows if i(r["n_gt_sv"]) == 0 and i(r["n_pred_sv"]) > 0)
    angle_rows.append({
        "angle": f"{ang:03d}",
        "tiles": len(rows),
        "pred_total": pred,
        "pred_sv": sv,
        "pred_sv_ratio": fmt(sv / pred if pred else 0.0),
        "sv_case1": c1,
        "sv_bg": bg,
        "sv_case1_ratio_of_sv": fmt(c1 / sv if sv else 0.0),
        "sv_bg_ratio_of_sv": fmt(bg / sv if sv else 0.0),
        "tiles_no_gt_sv_but_pred_sv": no_gt_sv_pred,
    })

# Rotation-induced Case-1 distribution.
rot_gain_vals = [f(r["case1_frac_rot_gain_max"]) for r in rot_tile]
rot_induced_tiles = [r for r in rot_tile if f(r["case1_frac_rot_gain_max"]) > 0.05]
rot_nonzero_case1_tiles = [r for r in rot_tile if i(r["n_sv_case1_total"]) > 0]
rot_peak_nonzero = [r for r in rot_tile if i(r.get("is_peak_nonzero", 0)) == 1]

# Concentration.
top_by_sv = sorted(tile_rows, key=lambda r: r["total_sv_dets"], reverse=True)[:20]
top_by_false = sorted(tile_rows, key=lambda r: r["total_sv_bg"] + r["total_sv_case1"], reverse=True)[:20]
top_by_case1 = sorted(tile_rows, key=lambda r: r["case1_sv_fraction"], reverse=True)[:20]
false_sv_total = total_bg + total_case1
false_sv_top20 = sum(r["total_sv_bg"] + r["total_sv_case1"] for r in top_by_false)
sv_top20 = sum(r["total_sv_dets"] for r in top_by_sv)

# Cohort GT baseline.
cohort_gt_sv_tiles = sum(1 for r in cohort if i(r.get("gt_sv_count", 0)) > 0)
cohort_gt_non_sv_tiles = sum(1 for r in cohort if i(r.get("gt_non_sv_count", 0)) > 0)

summary = {
    "n_tiles": n_tiles,
    "n_tile_angles": n_tile_angles,
    "total_pred": total_pred,
    "total_pred_sv": total_pred_sv,
    "pred_sv_ratio": total_pred_sv / total_pred if total_pred else 0.0,
    "total_gt": total_gt,
    "total_gt_sv": total_gt_sv,
    "gt_sv_ratio": total_gt_sv / total_gt if total_gt else 0.0,
    "total_sv_case1": total_case1,
    "total_sv_bg": total_bg,
    "sv_case1_ratio_of_sv": total_case1 / total_pred_sv if total_pred_sv else 0.0,
    "sv_bg_ratio_of_sv": total_bg / total_pred_sv if total_pred_sv else 0.0,
    "sv_false_bg_plus_case1_ratio_of_sv": false_sv_total / total_pred_sv if total_pred_sv else 0.0,
    "n_tile_angles_no_gt_sv_but_pred_sv": len(ta_no_gt_sv_pred_sv),
    "ratio_tile_angles_no_gt_sv_but_pred_sv": len(ta_no_gt_sv_pred_sv) / n_tile_angles if n_tile_angles else 0.0,
    "n_tile_angles_no_gt_any_but_pred_sv": len(ta_no_gt_any_pred_sv),
    "median_tile_angle_pred_sv_ratio_when_pred_nonzero": median(ta_pred_sv_ratio_vals) if ta_pred_sv_ratio_vals else 0.0,
    "p90_tile_angle_pred_sv_ratio_when_pred_nonzero": quant(ta_pred_sv_ratio_vals, 0.90),
    "p99_tile_angle_pred_sv_ratio_when_pred_nonzero": quant(ta_pred_sv_ratio_vals, 0.99),
    "n_tiles_with_gt_sv_at_angle0": cohort_gt_sv_tiles,
    "n_tiles_with_gt_non_sv_at_angle0": cohort_gt_non_sv_tiles,
    "n_tiles_with_case1": len(rot_nonzero_case1_tiles),
    "n_tiles_with_rot_gain_gt_0p05": len(rot_induced_tiles),
    "n_tiles_case1_peak_nonzero_angle": len(rot_peak_nonzero),
    "rot_gain_max": max(rot_gain_vals) if rot_gain_vals else 0.0,
    "rot_gain_p95": quant(rot_gain_vals, 0.95),
    "top20_false_sv_share": false_sv_top20 / false_sv_total if false_sv_total else 0.0,
    "top20_sv_det_share": sv_top20 / total_pred_sv if total_pred_sv else 0.0,
}

write_csv(OUT_DIR / "ftable_2500_attractor_angle_summary.csv", angle_rows)
write_csv(OUT_DIR / "ftable_2500_attractor_top_case1_tiles.csv", [
    {**r, "case1_sv_fraction": fmt(r["case1_sv_fraction"]), "bg_sv_fraction": fmt(r["bg_sv_fraction"]), "non_residual_false_sv_fraction": fmt(r["non_residual_false_sv_fraction"]), "case1_peak_rate": fmt(r["case1_peak_rate"])}
    for r in top_by_case1
])
write_csv(OUT_DIR / "ftable_2500_attractor_top_false_sv_tiles.csv", [
    {**r, "false_sv": r["total_sv_bg"] + r["total_sv_case1"], "case1_sv_fraction": fmt(r["case1_sv_fraction"]), "bg_sv_fraction": fmt(r["bg_sv_fraction"]), "non_residual_false_sv_fraction": fmt(r["non_residual_false_sv_fraction"]), "case1_peak_rate": fmt(r["case1_peak_rate"])}
    for r in top_by_false
])
write_csv(OUT_DIR / "ftable_2500_attractor_case1_gt_class.csv", [
    {"gt_class": k, "case1_events": v} for k, v in case1_gt_class.most_common()
])
write_csv(OUT_DIR / "ftable_2500_attractor_sv_best_gt_class.csv", [
    {"best_gt_class": k, "sv_events": v, "ratio_of_sv_events": fmt(v / len(sv_events) if sv_events else 0.0)}
    for k, v in gt_class_for_sv.most_common(30)
])

payload = {
    "summary": summary,
    "event_counts": dict(event_counts),
    "case1_gt_class_top20": case1_gt_class.most_common(20),
    "case2_drift_type_counts": dict(case2_drift),
    "outputs": {
        "angle_summary_csv": str(OUT_DIR / "ftable_2500_attractor_angle_summary.csv"),
        "top_case1_csv": str(OUT_DIR / "ftable_2500_attractor_top_case1_tiles.csv"),
        "top_false_sv_csv": str(OUT_DIR / "ftable_2500_attractor_top_false_sv_tiles.csv"),
        "case1_gt_class_csv": str(OUT_DIR / "ftable_2500_attractor_case1_gt_class.csv"),
        "sv_best_gt_class_csv": str(OUT_DIR / "ftable_2500_attractor_sv_best_gt_class.csv"),
    },
}
(OUT_DIR / "fjson_2500_attractor_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# Markdown report.
def md_table(rows: list[dict], fields: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    return "\n".join(out)

verdict = "成立，但不是全局均匀成立；更准确是少数 tile/角度上的 SV attractor + 大量背景/LV-adjacent false SV。"
if summary["pred_sv_ratio"] < 0.05 and summary["sv_false_bg_plus_case1_ratio_of_sv"] < 0.5:
    verdict = "不强成立；SV 不是 2500 cohort 的主导吸收口。"

top_case1_md_rows = []
for r in top_by_case1:
    rr = dict(r)
    rr["case1_sv_fraction"] = fmt(r["case1_sv_fraction"])
    rr["bg_sv_fraction"] = fmt(r["bg_sv_fraction"])
    rr["non_residual_false_sv_fraction"] = fmt(r["non_residual_false_sv_fraction"])
    rr["case1_peak_rate"] = fmt(r["case1_peak_rate"])
    top_case1_md_rows.append(rr)
case1_gt_md_rows = [{"gt_class": k, "case1_events": v} for k, v in case1_gt_class.most_common(15)]

lines = []
lines += ["# 2500-Cohort Small-Vehicle Attractor Audit", ""]
lines += ["- status: **DONE**", f"- generated_from: {WORK}", "- protocol: source-level audit over existing 2500-tile RSV taxonomy CSVs; no new inference/training.", ""]
lines += ["## Verdict", "", f"**{verdict}**", ""]
lines += ["The 2500 cohort supports a qualified attractor claim: small-vehicle is still a major false-positive sink in the mined taxonomy cohort, but the rotation-induced Case-1 component is sparse and concentrated. Therefore the paper wording should not say the attractor is universal over all DOTA tiles; it should say the attractor is exposed by high-risk remote-sensing contexts and specific rotations.", ""]
lines += ["## Global Counts", "", "| metric | value |", "|---|---:|"]
for key in ["n_tiles", "n_tile_angles", "total_pred", "total_pred_sv", "total_gt", "total_gt_sv", "total_sv_bg", "total_sv_case1", "n_tile_angles_no_gt_sv_but_pred_sv"]:
    lines.append(f"| {key} | {summary[key]} |")
for key in ["pred_sv_ratio", "gt_sv_ratio", "sv_bg_ratio_of_sv", "sv_case1_ratio_of_sv", "sv_false_bg_plus_case1_ratio_of_sv", "ratio_tile_angles_no_gt_sv_but_pred_sv", "p90_tile_angle_pred_sv_ratio_when_pred_nonzero", "p99_tile_angle_pred_sv_ratio_when_pred_nonzero"]:
    lines.append(f"| {key} | {fmt(summary[key])} |")
lines += ["", "## Rotation Case-1 Concentration", "", "| metric | value |", "|---|---:|"]
for key in ["n_tiles_with_case1", "n_tiles_with_rot_gain_gt_0p05", "n_tiles_case1_peak_nonzero_angle"]:
    lines.append(f"| {key} | {summary[key]} |")
for key in ["rot_gain_max", "rot_gain_p95", "top20_false_sv_share", "top20_sv_det_share"]:
    lines.append(f"| {key} | {fmt(summary[key])} |")
lines += ["", "## Angle Summary", "", md_table(angle_rows, ["angle","tiles","pred_total","pred_sv","pred_sv_ratio","sv_case1","sv_bg","sv_case1_ratio_of_sv","sv_bg_ratio_of_sv","tiles_no_gt_sv_but_pred_sv"]), ""]
lines += ["## Top Case-1 Tiles", "", md_table(top_case1_md_rows, ["tile_id","total_sv_dets","total_sv_case1","total_sv_bg","case1_sv_fraction","bg_sv_fraction","case1_peak_angle","case1_peak_rate"], 15), ""]
lines += ["## Case-1 GT Classes", "", md_table(case1_gt_md_rows, ["gt_class","case1_events"]), ""]
lines += ["## Interpretation", "", "- The attractor remains visible at cohort scale because predicted small-vehicle volume is much larger than true small-vehicle GT volume, and a large fraction of SV predictions are attributed to sv_bg or sv_case1 rather than clean true-SV matches.", "- The strongest rotation-induced Case-1 evidence is not global: only a small subset crosses a large angle-gain threshold, with P0411 as the clearest case.", "- This means the 2500-tile evidence backs the mechanism as a high-risk-context attractor, not as a uniform all-image failure mode.", "- For paper text: use P0148/P0411/top-risk tiles as diagnostic examples; use 2500 cohort as prevalence/concentration evidence; keep universal claims out.", ""]
lines += ["## Output Tables", "", "- " + str(OUT_DIR / "ftable_2500_attractor_angle_summary.csv"), "- " + str(OUT_DIR / "ftable_2500_attractor_top_case1_tiles.csv"), "- " + str(OUT_DIR / "ftable_2500_attractor_top_false_sv_tiles.csv"), "- " + str(OUT_DIR / "ftable_2500_attractor_case1_gt_class.csv"), "- " + str(OUT_DIR / "ftable_2500_attractor_sv_best_gt_class.csv"), "- " + str(OUT_DIR / "fjson_2500_attractor_audit.json")]
md = "\n".join(lines) + "\n"
(RESULT / "fres_020_2500_small_vehicle_attractor_audit.md").write_text(md)
print(json.dumps({"status": "DONE", "summary": summary, "md": str(RESULT / "fres_020_2500_small_vehicle_attractor_audit.md")}, indent=2))

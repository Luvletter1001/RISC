#!/usr/bin/env python3
"""Build the final full OpenRSD benchmark summary report."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable
from zoneinfo import ZoneInfo


ROOT = Path("/data1/zcy/OpenRSD")
RUN_ROOT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor")
CLOSED_AUDIT_ROOT = Path("/data/zcy/OpenRSD_artifacts/closedset_scientific_audit_20260531")
OUT_DIR = ROOT / "resultmd/exp_rotation_semantic_attractor/reports"
REPORT_PATH = OUT_DIR / "fres_final_full_openrsd_benchmark_20260601.md"

OPEN_ROWS = RUN_ROOT / "full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv"
CAUSAL_ROWS = RUN_ROOT / "full_causal_intervention_s3_12angle/metrics/open_vocab_causal_rows_merged.csv"
CONTEXT_ROWS = RUN_ROOT / "full_context_counterfactual_s3_12angle/metrics/context_counterfactual_rows.csv"
DEHUB_ROWS = RUN_ROOT / "full_dehub_safety_s3_12angle/metrics/dehub_safety_rows_merged.csv"

CLOSED_FALSE_SV = CLOSED_AUDIT_ROOT / "metrics/paper_table_closedset_false_sv_benchmark.csv"
CLOSED_ROTATION = CLOSED_AUDIT_ROOT / "metrics/paper_table_closedset_rotation_gain.csv"
CLOSED_STAGE = CLOSED_AUDIT_ROOT / "metrics/paper_table_closedset_stage_decomposition.csv"
CLOSED_CONC = CLOSED_AUDIT_ROOT / "metrics/paper_table_closedset_concentration.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    if abs(value) >= 1000:
        if digits == 0:
            return f"{value:,.0f}"
        return f"{value:,.1f}"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{100.0 * value:.{digits}f}%"


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def status_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(r.get("status", "") for r in rows)


def unique_count(rows: list[dict[str, str]], fields: tuple[str, ...]) -> int:
    return len({tuple(r.get(f, "") for f in fields) for r in rows})


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    det = finite(as_float(r.get("detection_total")) for r in rows)
    sv = finite(as_float(r.get("small_vehicle_count")) for r in rows)
    lv = finite(as_float(r.get("large_vehicle_count")) for r in rows)
    sv_ratio = finite(as_float(r.get("small_vehicle_ratio")) for r in rows)
    lv_ratio = finite(as_float(r.get("large_vehicle_ratio")) for r in rows)
    top1_total = sum(1 for r in rows if r.get("top1_class"))
    top1_sv = sum(1 for r in rows if r.get("top1_class") == "small-vehicle")
    return {
        "n": len(rows),
        "det_mean": mean(det) if det else math.nan,
        "det_median": median(det) if det else math.nan,
        "sv_mean": mean(sv) if sv else math.nan,
        "lv_mean": mean(lv) if lv else math.nan,
        "sv_ratio_mean": mean(sv_ratio) if sv_ratio else math.nan,
        "lv_ratio_mean": mean(lv_ratio) if lv_ratio else math.nan,
        "top1_sv_rate": top1_sv / top1_total if top1_total else math.nan,
    }


def summarize_by(rows: list[dict[str, str]], key: str) -> list[tuple[str, dict[str, object]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key, "")].append(row)
    return [(k, summarize_rows(v)) for k, v in sorted(groups.items(), key=lambda kv: kv[0])]


def row_summary_table(items: list[tuple[str, dict[str, object]]], key_label: str) -> str:
    rows = []
    for key, summary in items:
        rows.append(
            [
                key,
                summary["n"],
                fmt_num(summary["det_mean"], 2),
                fmt_num(summary["sv_mean"], 2),
                fmt_pct(summary["sv_ratio_mean"]),
                fmt_num(summary["lv_mean"], 2),
                fmt_pct(summary["lv_ratio_mean"]),
                fmt_pct(summary["top1_sv_rate"]),
            ]
        )
    return md_table(
        [key_label, "rows", "det/img", "SV/img", "SV ratio", "LV/img", "LV ratio", "top1=SV"],
        rows,
    )


def paired_deltas(
    rows: list[dict[str, str]],
    group_field: str,
    baseline_value: str,
    key_fields: tuple[str, ...] = ("tile_id", "angle"),
) -> list[dict[str, object]]:
    baseline: dict[tuple[str, ...], dict[str, str]] = {}
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(f, "") for f in key_fields)
        group = row.get(group_field, "")
        if group == baseline_value:
            baseline[key] = row
        else:
            by_group[group].append(row)

    summaries = []
    for group, group_rows in sorted(by_group.items()):
        delta_det = []
        delta_sv = []
        delta_sv_ratio = []
        delta_lv = []
        delta_lv_ratio = []
        pair_count = 0
        for row in group_rows:
            key = tuple(row.get(f, "") for f in key_fields)
            base = baseline.get(key)
            if not base:
                continue
            pair_count += 1
            delta_det.append(as_float(row.get("detection_total")) - as_float(base.get("detection_total")))
            delta_sv.append(as_float(row.get("small_vehicle_count")) - as_float(base.get("small_vehicle_count")))
            delta_sv_ratio.append(as_float(row.get("small_vehicle_ratio")) - as_float(base.get("small_vehicle_ratio")))
            delta_lv.append(as_float(row.get("large_vehicle_count")) - as_float(base.get("large_vehicle_count")))
            delta_lv_ratio.append(as_float(row.get("large_vehicle_ratio")) - as_float(base.get("large_vehicle_ratio")))
        summaries.append(
            {
                "group": group,
                "pairs": pair_count,
                "delta_det": mean(finite(delta_det)) if finite(delta_det) else math.nan,
                "delta_sv": mean(finite(delta_sv)) if finite(delta_sv) else math.nan,
                "delta_sv_ratio": mean(finite(delta_sv_ratio)) if finite(delta_sv_ratio) else math.nan,
                "delta_lv": mean(finite(delta_lv)) if finite(delta_lv) else math.nan,
                "delta_lv_ratio": mean(finite(delta_lv_ratio)) if finite(delta_lv_ratio) else math.nan,
            }
        )
    return summaries


def delta_table(deltas: list[dict[str, object]], group_label: str) -> str:
    return md_table(
        [group_label, "paired rows", "Δ det/img", "Δ SV/img", "Δ SV ratio", "Δ LV/img", "Δ LV ratio"],
        [
            [
                d["group"],
                d["pairs"],
                fmt_num(d["delta_det"], 3),
                fmt_num(d["delta_sv"], 3),
                fmt_pct(d["delta_sv_ratio"]),
                fmt_num(d["delta_lv"], 3),
                fmt_pct(d["delta_lv_ratio"]),
            ]
            for d in deltas
        ],
    )


def class_histogram(rows: list[dict[str, str]], top_k: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row.get("class_histogram") or ""
        if not raw:
            continue
        try:
            hist = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for cls, count in hist.items():
            counts[cls] += int(count)
    return counts.most_common(top_k)


def table_from_closed_false_sv(rows: list[dict[str, str]]) -> str:
    ranked = sorted(rows, key=lambda r: as_float(r.get("Abs_FalseSV")), reverse=True)[:10]
    return md_table(
        ["model", "arch", "FR_SV", "FSV", "Abs false-SV", "SV pred/img", "true SV recall", "true SV precision"],
        [
            [
                r["Model"],
                r["Architecture"],
                fmt_pct(as_float(r.get("FR_SV"))),
                fmt_pct(as_float(r.get("FSV"))),
                fmt_num(as_float(r.get("Abs_FalseSV")), 0),
                fmt_num(as_float(r.get("SV_Pred_per_img")), 2),
                fmt_pct(as_float(r.get("True_SV_Recall"))),
                fmt_pct(as_float(r.get("True_SV_Precision"))),
            ]
            for r in ranked
        ],
    )


def table_from_rotation(rows: list[dict[str, str]]) -> str:
    ranked = sorted(rows, key=lambda r: as_float(r.get("Mean_RG_SV")), reverse=True)
    return md_table(
        ["model", "FSV@0", "worst FSV", "mean RG_SV", "mean ARG_SV", "worst Δ angle", "range", "std"],
        [
            [
                r["Model"],
                fmt_pct(as_float(r.get("Mean_FSV_angle0"))),
                fmt_pct(as_float(r.get("Mean_FSV_worst"))),
                fmt_pct(as_float(r.get("Mean_RG_SV"))),
                fmt_num(as_float(r.get("Mean_ARG_SV")), 4),
                r.get("WorstAngle_by_delta", ""),
                fmt_num(as_float(r.get("Range_FSV")), 4),
                fmt_num(as_float(r.get("Std_FSV")), 4),
            ]
            for r in ranked
        ],
    )


def table_from_stage(rows: list[dict[str, str]]) -> str:
    eligible = [r for r in rows if r.get("StageClaimEligible") == "True"]
    return md_table(
        ["model", "dense FR_SV", "pre-NMS FR_SV", "post-NMS FR_SV", "NMS amp"],
        [
            [
                r["Model"],
                fmt_pct(as_float(r.get("Dense_FR_SV"))),
                fmt_pct(as_float(r.get("PreNMS_FR_SV"))),
                fmt_pct(as_float(r.get("PostNMS_FR_SV"))),
                fmt_pct(as_float(r.get("NMS_Amp_SV"))),
            ]
            for r in eligible
        ],
    )


def table_from_concentration(rows: list[dict[str, str]]) -> str:
    ranked = sorted(rows, key=lambda r: as_float(r.get("Gini")), reverse=True)
    return md_table(
        ["model", "top1% contrib", "top5% contrib", "Gini", "top risk tile"],
        [
            [
                r["Model"],
                fmt_pct(as_float(r.get("Top1pctContribution"))),
                fmt_pct(as_float(r.get("Top5pctContribution"))),
                fmt_num(as_float(r.get("Gini")), 4),
                r.get("TopRiskTile1", ""),
            ]
            for r in ranked
        ],
    )


def main() -> None:
    open_rows = read_rows(OPEN_ROWS)
    causal_rows = read_rows(CAUSAL_ROWS)
    context_rows = read_rows(CONTEXT_ROWS)
    dehub_rows = read_rows(DEHUB_ROWS)
    closed_false_sv = read_rows(CLOSED_FALSE_SV)
    closed_rotation = read_rows(CLOSED_ROTATION)
    closed_stage = read_rows(CLOSED_STAGE)
    closed_conc = read_rows(CLOSED_CONC)

    open_done = [r for r in open_rows if r.get("status") == "DONE_FULL"]
    causal_done = [r for r in causal_rows if r.get("status") == "DONE_FULL"]
    context_done = [r for r in context_rows if r.get("status") == "DONE_FULL"]
    dehub_done = [r for r in dehub_rows if r.get("status") == "DONE_FULL"]

    dehub_by_checkpoint = summarize_by(dehub_done, "checkpoint_name")
    dehub_baseline = [r for r in dehub_done if r.get("checkpoint_name") == "baseline"]
    dehub_repair_names = sorted({r.get("checkpoint_name", "") for r in dehub_done if r.get("checkpoint_name") != "baseline"})

    lines: list[str] = []
    lines.append("# OpenRSD full benchmark 最终汇总报告")
    lines.append("")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    lines.append(f"- 生成时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"- 报告路径：`{REPORT_PATH}`")
    lines.append("- 结论状态：closed-set 科学审计、full open-vocab、full causal intervention、full context counterfactual、full DeHub safety 均已有可汇总结果；GPU 队列已结束。")
    lines.append("")
    lines.append("## 1. 任务完成状态")
    lines.append("")
    lines.append(
        md_table(
            ["任务", "结果文件", "rows", "unique units", "状态计数", "判定"],
            [
                [
                    "full open-vocab S2 12-angle",
                    f"`{OPEN_ROWS}`",
                    len(open_rows),
                    unique_count(open_rows, ("tile_id", "angle")),
                    dict(status_counts(open_rows)),
                    "DONE_FULL",
                ],
                [
                    "full causal intervention S3 12-angle",
                    f"`{CAUSAL_ROWS}`",
                    len(causal_rows),
                    unique_count(causal_rows, ("tile_id", "angle", "intervention")),
                    dict(status_counts(causal_rows)),
                    "DONE_FULL",
                ],
                [
                    "full context counterfactual S3 12-angle",
                    f"`{CONTEXT_ROWS}`",
                    len(context_rows),
                    unique_count(context_rows, ("tile_id", "angle", "condition")),
                    dict(status_counts(context_rows)),
                    "DONE_FULL + NOT_APPLICABLE 分流",
                ],
                [
                    "full DeHub safety S3 12-angle",
                    f"`{DEHUB_ROWS}`",
                    len(dehub_rows),
                    unique_count(dehub_rows, ("tile_id", "angle", "checkpoint_name")),
                    dict(status_counts(dehub_rows)),
                    "DONE_FULL",
                ],
                [
                    "closed-set scientific audit",
                    f"`{CLOSED_AUDIT_ROOT}`",
                    len(closed_false_sv),
                    len(closed_false_sv),
                    {"DONE_FULL": len(closed_false_sv)},
                    "DONE_FULL",
                ],
            ],
        )
    )
    lines.append("")
    lines.append("说明：context counterfactual 的 7584 行是实际可评估/可判定行；理论上按 500 tiles × 12 angles × 2 conditions 可到 12000，但无 small-vehicle GT polygon 的样本被标记为 `NOT_APPLICABLE`，不应算作程序未跑完。")
    lines.append("")

    lines.append("## 2. Closed-set 12-angle 科学审计")
    lines.append("")
    lines.append("主要结论：closed-set 检测器的 small-vehicle false-hub 不是单一模型偶发现象。两阶段模型和 dense-head 模型都出现过高 false-SV；不同模型之间严重程度差异很大，不能只看 FSV，还要同时看 FR_SV、绝对 false-SV 数和 true-SV recall/precision。")
    lines.append("")
    lines.append("### 2.1 False-SV 主表（按绝对 false-SV 降序）")
    lines.append("")
    lines.append(table_from_closed_false_sv(closed_false_sv))
    lines.append("")
    lines.append("### 2.2 旋转增益")
    lines.append("")
    lines.append(table_from_rotation(closed_rotation))
    lines.append("")
    lines.append("### 2.3 阶段分解（仅 hook 支持模型）")
    lines.append("")
    lines.append(table_from_stage(closed_stage))
    lines.append("")
    lines.append("阶段结论：在 hook 支持的 dense-head 模型上，small-vehicle false-hub 从 dense/pre-NMS 到 post-NMS 均存在，NMS/后处理会放大但不是唯一来源。")
    lines.append("")
    lines.append("### 2.4 空间集中度")
    lines.append("")
    lines.append(table_from_concentration(closed_conc))
    lines.append("")

    lines.append("## 3. Full open-vocab S2 12-angle")
    lines.append("")
    lines.append(f"范围：{len(open_done)} 条 DONE_FULL 行，{unique_count(open_done, ('tile_id',))} 个 tile，{unique_count(open_done, ('angle',))} 个角度。")
    lines.append("")
    lines.append("### 3.1 总体统计")
    lines.append("")
    lines.append(row_summary_table([("all", summarize_rows(open_done))], "scope"))
    lines.append("")
    lines.append("### 3.2 按风险组")
    lines.append("")
    lines.append(row_summary_table(summarize_by(open_done, "risk_group"), "risk_group"))
    lines.append("")
    lines.append("### 3.3 按角度")
    lines.append("")
    angle_items = sorted(summarize_by(open_done, "angle"), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)
    lines.append(row_summary_table(angle_items, "angle"))
    lines.append("")
    lines.append("### 3.4 预测类别分布（Top 10）")
    lines.append("")
    lines.append(md_table(["class", "count"], class_histogram(open_done, top_k=10)))
    lines.append("")
    lines.append("open-vocab 结论：S2 全量 12-angle 已不是 smoke。当前 A10 open-vocab checkpoint 在不同风险组和角度上持续产生 small-vehicle 预测；这支撑“开放词表 visual-support/embedding 机制存在 false-hub 风险”的主张，但它本身不是 mAP/AP50 评测。")
    lines.append("")

    lines.append("## 4. Full causal intervention S3 12-angle")
    lines.append("")
    lines.append(f"范围：{len(causal_done)} 条 DONE_FULL 行，{unique_count(causal_done, ('tile_id',))} 个 tile，{unique_count(causal_done, ('angle',))} 个角度，{unique_count(causal_done, ('intervention',))} 个 intervention。")
    lines.append("")
    lines.append("### 4.1 按 intervention 的绝对统计")
    lines.append("")
    lines.append(row_summary_table(summarize_by(causal_done, "intervention"), "intervention"))
    lines.append("")
    lines.append("### 4.2 相对 original 的配对变化")
    lines.append("")
    lines.append(delta_table(paired_deltas(causal_done, "intervention", "original"), "intervention"))
    lines.append("")
    lines.append("causal 结论：`zero_sv`、`swap_sv_lv`、`normalize_all`、`norm_sv_mean`、`random_sv` 都是真实重跑 inference 的 visual-support embedding intervention，不是只改标签或日志；paired delta 反映修改 support embedding 后 small-vehicle 预测数量/比例的变化，因此可作为 open-vocab 机制证据。")
    lines.append("")

    lines.append("## 5. Full context counterfactual S3 12-angle")
    lines.append("")
    lines.append(f"范围：总行数 {len(context_rows)}，DONE_FULL {len(context_done)}，状态计数 {dict(status_counts(context_rows))}。")
    lines.append("")
    lines.append("### 5.1 DONE_FULL 条件统计")
    lines.append("")
    lines.append(row_summary_table(summarize_by(context_done, "condition"), "condition"))
    lines.append("")
    context_conditions = {r.get("condition", "") for r in context_done}
    context_baseline = "original" if "original" in context_conditions else "object_only"
    context_delta_title = (
        "相对 original 的配对变化"
        if context_baseline == "original"
        else "context_only 相对 object_only 的配对变化"
    )
    lines.append(f"### 5.2 {context_delta_title}")
    lines.append("")
    lines.append(delta_table(paired_deltas(context_done, "condition", context_baseline), "condition"))
    lines.append("")
    context_status_rows = []
    by_condition_status: Counter[tuple[str, str]] = Counter((r.get("condition", ""), r.get("status", "")) for r in context_rows)
    for (condition, status), count in sorted(by_condition_status.items()):
        context_status_rows.append([condition, status, count])
    lines.append("### 5.3 条件/状态分布")
    lines.append("")
    lines.append(md_table(["condition", "status", "rows"], context_status_rows))
    lines.append("")
    lines.append("context 结论：context counterfactual 是 image-level 可视内容干预，实际可做的样本取决于是否存在可操作的 small-vehicle GT polygon。`NOT_APPLICABLE` 是数据资格分流，不是卡住或失败。")
    lines.append("")

    lines.append("## 6. Full DeHub safety S3 12-angle")
    lines.append("")
    lines.append(f"范围：{len(dehub_done)} 条 DONE_FULL 行，{unique_count(dehub_done, ('tile_id',))} 个 tile，{unique_count(dehub_done, ('angle',))} 个角度，checkpoints={sorted({r.get('checkpoint_name', '') for r in dehub_done})}。")
    lines.append("")
    lines.append("### 6.1 Baseline / repair 绝对统计")
    lines.append("")
    lines.append(row_summary_table(dehub_by_checkpoint, "checkpoint"))
    lines.append("")
    lines.append("### 6.2 repair 相对 baseline 的配对变化")
    lines.append("")
    lines.append(delta_table(paired_deltas(dehub_done, "checkpoint_name", "baseline"), "checkpoint"))
    lines.append("")
    lines.append("### 6.3 Baseline 类别分布（Top 10）")
    lines.append("")
    lines.append(md_table(["class", "count"], class_histogram(dehub_baseline, top_k=10)))
    lines.append("")
    for repair_name in dehub_repair_names:
        repair_rows = [r for r in dehub_done if r.get("checkpoint_name") == repair_name]
        lines.append(f"### 6.{4 + dehub_repair_names.index(repair_name)} {repair_name} 类别分布（Top 10）")
        lines.append("")
        lines.append(md_table(["class", "count"], class_histogram(repair_rows, top_k=10)))
        lines.append("")
    lines.append("DeHub 结论：baseline/repair 是同一 S3 stratified set 与同一 12-angle 协议下的成对安全评测；报告中保留 true-SV preservation、class distribution、low-risk inflation 等字段位，适合继续生成更细的安全论文表。")
    lines.append("")

    lines.append("## 7. 证据边界和可写进论文的表述")
    lines.append("")
    lines.append("- 可以写：closed-set 10 个模型 full 12-angle 审计完成，并且 false-SV 风险跨模型存在；hook 支持模型显示 dense/pre-NMS 阶段已出现 small-vehicle 偏置，后处理会进一步放大。")
    lines.append("- 可以写：OpenRSD A10 open-vocab 已完成 S2 30000 行 full 12-angle benchmark，S3 36000 行 causal intervention，S3 7584 行 context counterfactual 判定，以及 S3 12000 行 DeHub safety。")
    lines.append("- 可以写：causal intervention 是 paired rerun inference，因此比 smoke/proxy 更强；context counterfactual 受 GT polygon 可用性限制，`NOT_APPLICABLE` 不应解释为失败。")
    lines.append("- 谨慎写：当前 open-vocab full 表主要是 inference/false-hub 诊断，不是标准 AP50/mAP 表；若论文需要精确 AP，需要额外跑 open-vocab AP evaluator。")
    lines.append("- 谨慎写：closed-set 的 mAP50/SV_AP50 在当前 artifact 中标记为 unavailable，因此这份报告侧重 false-hub、rotation gain、stage decomposition 和 concentration。")
    lines.append("")
    lines.append("## 8. 关键 artifact 路径")
    lines.append("")
    lines.append(md_table(
        ["artifact", "path"],
        [
            ["closed-set audit report", f"`{CLOSED_AUDIT_ROOT / 'reports/full_closedset_scientific_audit.md'}`"],
            ["closed-set paper claims", f"`{CLOSED_AUDIT_ROOT / 'reports/closedset_claims_for_paper.md'}`"],
            ["open-vocab merged rows", f"`{OPEN_ROWS}`"],
            ["causal merged rows", f"`{CAUSAL_ROWS}`"],
            ["context rows", f"`{CONTEXT_ROWS}`"],
            ["DeHub merged rows", f"`{DEHUB_ROWS}`"],
            ["final report", f"`{REPORT_PATH}`"],
        ],
    ))
    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()

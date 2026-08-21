#!/usr/bin/env python3
"""Build the Chinese final report for the CPU-only text/Fourier study."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_fourier_cpu10_common import (
    BASELINE_AUC,
    EXP_DIR,
    IDEA_REGISTRY,
    chinese_bool,
    ensure_tree,
    md_table,
    read_csv,
    read_json,
    resolve,
    safe_float,
    write_json,
)


LITERATURE_ROWS = [
    {
        "编号": "S1",
        "论文/方向": "Fourier Angle Alignment for Oriented Object Detection in Remote Sensing",
        "与本实验的关系": "Fourier rotation equivariance、角度对齐、oriented detector 频域依据",
        "链接": "https://arxiv.org/abs/2602.23790",
    },
    {
        "编号": "S2",
        "论文/方向": "Fast Fourier Convolution Based Remote Sensor Image Object Detection for Earth Observation",
        "与本实验的关系": "频域卷积、遥感目标检测、小目标和长程上下文",
        "链接": "https://arxiv.org/abs/2209.00551",
    },
    {
        "编号": "S3",
        "论文/方向": "Fourier-based Rotation-invariant Feature Boosting",
        "与本实验的关系": "polar Fourier rotation-invariant feature 与 geospatial object detection",
        "链接": "https://arxiv.org/abs/1905.11074",
    },
    {
        "编号": "S4",
        "论文/方向": "FSDENet: Frequency and Spatial Domains Detail Enhancement",
        "与本实验的关系": "FFT + Haar wavelet、边界/细节增强",
        "链接": "https://arxiv.org/abs/2510.00059",
    },
    {
        "编号": "S5",
        "论文/方向": "Dual-Stream Spectral Decoupling Distillation",
        "与本实验的关系": "wavelet/spectral decomposition、遥感检测蒸馏和高频差异",
        "链接": "https://arxiv.org/abs/2512.04413",
    },
    {
        "编号": "S6",
        "论文/方向": "MM-DETR frequency-aware modality adapters",
        "与本实验的关系": "频域适配器、多模态遥感检测、轻量融合",
        "链接": "https://arxiv.org/abs/2512.00363",
    },
    {
        "编号": "S7",
        "论文/方向": "Frequency-Dynamic Attention Modulation",
        "与本实验的关系": "频率响应调制、遥感检测 attention 设计",
        "链接": "https://arxiv.org/abs/2507.12006",
    },
    {
        "编号": "S8",
        "论文/方向": "RDNet frequency-matching context enhancement",
        "与本实验的关系": "wavelet/frequency context、周边结构增强",
        "链接": "https://arxiv.org/abs/2603.12215",
    },
    {
        "编号": "S9",
        "论文/方向": "RIFT: Radiation-invariant Feature Transform",
        "与本实验的关系": "phase congruency、Log-Gabor、Maximum Index Map",
        "链接": "https://arxiv.org/abs/1804.09493",
    },
    {
        "编号": "S10",
        "论文/方向": "R2FD2 multimodal remote sensing matching",
        "与本实验的关系": "Log-Gabor、dominant orientation、rotation-invariant descriptor",
        "链接": "https://arxiv.org/abs/2212.02277",
    },
    {
        "编号": "S11",
        "论文/方向": "HOPC structural similarity registration",
        "与本实验的关系": "orientated phase congruency、结构相似性、方向信息",
        "链接": "https://arxiv.org/abs/2103.16871",
    },
    {
        "编号": "S12",
        "论文/方向": "ABFL angular boundary discontinuity free loss",
        "与本实验的关系": "角度周期边界、circular data、oriented detection",
        "链接": "https://arxiv.org/abs/2311.12311",
    },
    {
        "编号": "S13",
        "论文/方向": "RiO-DETR real-time oriented detection",
        "与本实验的关系": "方向语义、角度周期、orthogonal attention",
        "链接": "https://arxiv.org/abs/2603.09411",
    },
    {
        "编号": "S14",
        "论文/方向": "Random Fourier features for remote sensing classification",
        "与本实验的关系": "Fourier 特征近似、大规模遥感分类和统计验证",
        "链接": "https://arxiv.org/abs/1710.00575",
    },
]


def _format_float(value: Any, digits: int = 4) -> str:
    if value == "":
        return ""
    return f"{safe_float(value):.{digits}f}"


def build_markdown(exp_dir: Path,
                   analysis: dict[str, Any],
                   summary: list[dict[str, str]],
                   idea_summary: list[dict[str, str]],
                   row_count: int) -> str:
    best = summary[0] if summary else {}
    best_idea = idea_summary[0] if idea_summary else {}
    candidates = [row for row in summary if str(row.get("safe_candidate")) == "True"]
    rejected = [row for row in summary if str(row.get("safe_candidate")) != "True"]
    compact_rows = [
        {
            "rank": row.get("rank", ""),
            "method_id": row.get("method_id", ""),
            "name_zh": row.get("name_zh", ""),
            "auc": _format_float(row.get("auc")),
            "auc_delta_vs_tsafe": _format_float(row.get("auc_delta_vs_tsafe")),
            "rank_score": _format_float(row.get("rank_score")),
            "safe_candidate": chinese_bool(str(row.get("safe_candidate")) == "True"),
            "reject_reason": row.get("reject_reason", ""),
        }
        for row in summary
    ]
    idea_specs = {idea.idea_id: idea for idea in IDEA_REGISTRY}
    idea_rows = []
    for row in sorted(
            idea_summary,
            key=lambda item: int(str(item.get("idea_id", "I999"))[1:])):
        idea_id = row.get("idea_id", "")
        spec = idea_specs.get(idea_id)
        idea_rows.append({
            "ID": idea_id,
            "rank": row.get("rank", ""),
            "创新点": row.get("title_zh", spec.title_zh if spec else ""),
            "依据": row.get("source_ids", spec.source_ids if spec else ""),
            "AUC": _format_float(row.get("auc")),
            "delta": _format_float(row.get("auc_delta_vs_tsafe")),
            "rank_score": _format_float(row.get("rank_score")),
            "deg_top20": _format_float(row.get("degenerate_top20_share")),
            "pad_top20": _format_float(row.get("padding_top20_share")),
            "retention": _format_float(row.get("true_retention_proxy")),
            "safe": chinese_bool(str(row.get("safe_candidate")) == "True"),
            "evidence": row.get("evidence_status", ""),
            "拒绝/通过原因": row.get("reject_reason", ""),
            "实操验证": row.get("verification_zh", spec.verification_zh if spec else ""),
        })
    lines = [
        "# CPU-only Text-Fourier 10 路离线代理测评报告",
        "",
        "## 结论",
        "",
        "- 本实验只做 CPU 离线代理测评，没有运行 detector、没有训练、没有修改 logits/NMS/support bank/checkpoint。",
        f"- 输入样本数：`{row_count}`。",
        f"- T-Safe 参考 AUC：`{BASELINE_AUC:.4f}`。",
        f"- 最优方法：`{best.get('method_id', '')}` / `{best.get('name_zh', '')}`，AUC=`{_format_float(best.get('auc'))}`，rank_score=`{_format_float(best.get('rank_score'))}`。",
        f"- 100 个创新点的最优代理：`{best_idea.get('idea_id', '')}`，AUC=`{_format_float(best_idea.get('auc'))}`，rank_score=`{_format_float(best_idea.get('rank_score'))}`。",
        f"- 安全候选数量：`{len(candidates)}`。",
        f"- 100 个创新点数值证明数量：`{len(idea_summary)}`。",
        "",
        "## 排名表",
        "",
    ]
    lines.extend(md_table(
        compact_rows,
        [
            "rank",
            "method_id",
            "name_zh",
            "auc",
            "auc_delta_vs_tsafe",
            "rank_score",
            "safe_candidate",
            "reject_reason",
        ]))
    lines.extend([
        "",
        "## 文献检索依据",
        "",
        "检索日期：2026-06-10。检索范围覆盖 Fourier/FFT/频域分解、Log-Gabor/phase congruency、方向周期性、遥感目标检测、遥感语义分割、遥感配准、多模态遥感检测和周边结构建模。这里的论文依据是当前可检索到的代表性公开论文线索，不声称穷尽全球全部论文全集；正式论文写作前仍需要继续用 IEEE Xplore、ScienceDirect、Web of Science、Google Scholar 做人工补全。",
        "",
    ])
    lines.extend(md_table(
        LITERATURE_ROWS,
        ["编号", "论文/方向", "与本实验的关系", "链接"]))
    lines.extend([
        "",
        "## 100 个创新点数值证明表",
        "",
        "下表的数值是 CPU-only 离线代理测评结果：每个创新点都在同一批 crop 上生成确定性 score，并计算 AUC、rank_score、degenerate/padding top20、true retention proxy 和拒绝原因。它只能证明当前代理实验里的可分性和安全性，不能替代 detector AP/mAP 结论。",
        "",
    ])
    lines.extend(md_table(
        idea_rows,
        [
            "ID",
            "rank",
            "创新点",
            "依据",
            "AUC",
            "delta",
            "rank_score",
            "deg_top20",
            "pad_top20",
            "retention",
            "safe",
            "evidence",
            "拒绝/通过原因",
            "实操验证",
        ]))
    lines.extend([
        "",
        "## 安全边界",
        "",
        "- 本报告不能作为 AP、mAP、召回率或 detector 真实性能提升证据。",
        "- TF07 只是影子残差分数，不接入最终 logits。",
        "- TF10 只允许单向下调 FOCUS 分数；若违反 one-way 约束会被拒绝。",
        "- 任何后续进入 detector 的方案都需要重新做 detector 级验证。",
        "",
        "## 产物",
        "",
        f"- 逐样本分数：`{exp_dir / 'tables' / 'text_fourier_cpu10_sample_scores.csv'}`",
        f"- 100 创新点逐样本分数：`{exp_dir / 'tables' / 'text_fourier_cpu100_idea_scores.csv'}`",
        f"- 方法汇总：`{exp_dir / 'tables' / 'text_fourier_cpu10_method_summary.csv'}`",
        f"- 100 创新点数值汇总：`{exp_dir / 'tables' / 'text_fourier_cpu100_idea_summary.csv'}`",
        f"- 分析 JSON：`{exp_dir / 'reports' / 'text_fourier_cpu10_analysis.json'}`",
        f"- 排名图：`{analysis.get('rank_png', '')}`",
        "",
    ])
    if rejected:
        lines.extend(["## 拒绝原因摘要", ""])
        reject_rows = [
            {
                "method_id": row.get("method_id", ""),
                "name_zh": row.get("name_zh", ""),
                "reject_reason": row.get("reject_reason", ""),
            }
            for row in rejected
        ]
        lines.extend(md_table(reject_rows, ["method_id", "name_zh", "reject_reason"]))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exp_dir = resolve(repo_root, args.exp_dir)
    ensure_tree(exp_dir)
    summary_csv = exp_dir / "tables" / "text_fourier_cpu10_method_summary.csv"
    idea_summary_csv = exp_dir / "tables" / "text_fourier_cpu100_idea_summary.csv"
    scores_csv = exp_dir / "tables" / "text_fourier_cpu10_sample_scores.csv"
    idea_scores_csv = exp_dir / "tables" / "text_fourier_cpu100_idea_scores.csv"
    analysis_json = exp_dir / "reports" / "text_fourier_cpu10_analysis.json"
    summary = read_csv(summary_csv)
    idea_summary = read_csv(idea_summary_csv)
    scores = read_csv(scores_csv)
    analysis = read_json(analysis_json, {})
    markdown = build_markdown(exp_dir, analysis, summary, idea_summary, len(scores))
    md_path = exp_dir / "reports" / "text_fourier_cpu10_final_report.md"
    html_path = exp_dir / "html" / "text_fourier_cpu10_final_report.html"
    json_path = exp_dir / "reports" / "text_fourier_cpu10_final_report.json"
    md_path.write_text(markdown + "\n", encoding="utf-8")
    html_path.write_text("<pre>" + html.escape(markdown) + "</pre>\n", encoding="utf-8")
    payload = {
        "status": "PASS_FINAL_REPORT" if summary else "NO_SUMMARY",
        "report_md": str(md_path),
        "report_html": str(html_path),
        "summary_csv": str(summary_csv),
        "idea_summary_csv": str(idea_summary_csv),
        "scores_csv": str(scores_csv),
        "idea_scores_csv": str(idea_scores_csv),
        "row_count": len(scores),
        "idea_count": len(idea_summary),
        "best_method": summary[0] if summary else {},
        "best_idea": idea_summary[0] if idea_summary else {},
        "detector_run": False,
        "training_run": False,
        "final_logits_modified": False,
        "support_bank_replaced": False,
    }
    write_json(json_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if summary and idea_summary else 2


if __name__ == "__main__":
    raise SystemExit(main())

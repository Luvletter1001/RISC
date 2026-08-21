#!/usr/bin/env python3
"""Summarize completed DOTA2 experiments without requiring GPUs."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "work_dirs" / "analysis_reports" / "gpu_free_20260430"

CLASSES = [
    "airport",
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "container-crane",
    "ground-track-field",
    "harbor",
    "helicopter",
    "helipad",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]

LOGS = {
    "lsknet_baseline_fullinit": ROOT
    / "work_dirs"
    / "lsknet_dotav2_ss_orcnn_bs2_fullinit"
    / "20260428_142330"
    / "20260428_142330.log",
    "faahead_lsknet_lr1e4_noema": ROOT
    / "work_dirs"
    / "faahead_dotav2_ss_lsknet_bs2_lr1e4_noema"
    / "20260428_161338"
    / "20260428_161338.log",
}

CONFIGS = {
    "save_best": ROOT
    / "M_configs"
    / "G02_Baselines"
    / "Data1_DOTA2"
    / "G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest.py",
    "save_best_ema": ROOT
    / "M_configs"
    / "G02_Baselines"
    / "Data1_DOTA2"
    / "G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest_EMA.py",
    "save_best_earlylr": ROOT
    / "M_configs"
    / "G02_Baselines"
    / "Data1_DOTA2"
    / "G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest_EarlyLR.py",
}

ROW_RE = re.compile(
    r"^\|\s*(?P<class>[^|]+?)\s*\|\s*(?P<gts>\d+)\s*\|\s*"
    r"(?P<dets>\d+)\s*\|\s*(?P<recall>[0-9.]+)\s*\|\s*(?P<ap>[0-9.]+)\s*\|"
)
MAP_ROW_RE = re.compile(r"^\|\s*mAP\s*\|.*\|\s*(?P<table_map>[0-9.]+)\s*\|")
EPOCH_RE = re.compile(
    r"Epoch\(val\)\s*\[(?P<epoch>\d+)\].*dota/mAP:\s*(?P<map>[0-9.]+)"
    r"\s+dota/AP50:\s*(?P<ap50>[0-9.]+)"
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_eval_tables(log_path: Path, experiment: str) -> list[dict[str, object]]:
    lines = log_path.read_text(errors="replace").splitlines()
    rows: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    table_map = None
    for idx, line in enumerate(lines):
        row_match = ROW_RE.match(line)
        if row_match:
            item = row_match.groupdict()
            pending.append(
                {
                    "experiment": experiment,
                    "log_path": rel(log_path),
                    "class": item["class"],
                    "gts": int(item["gts"]),
                    "dets": int(item["dets"]),
                    "recall": float(item["recall"]),
                    "ap": float(item["ap"]),
                }
            )
            continue

        map_match = MAP_ROW_RE.match(line)
        if map_match:
            table_map = float(map_match.group("table_map"))
            continue

        epoch_match = EPOCH_RE.search(line)
        if epoch_match and pending:
            epoch = int(epoch_match.group("epoch"))
            overall_map = float(epoch_match.group("map"))
            overall_ap50 = float(epoch_match.group("ap50"))
            for item in pending:
                item["epoch"] = epoch
                item["overall_map"] = overall_map
                item["overall_ap50"] = overall_ap50
                item["table_map"] = table_map
                item["line_no"] = idx + 1
                rows.append(item)
            pending = []
            table_map = None
    return rows


def box_stats(points: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    shoelace = (
        x1 * y2
        + x2 * y3
        + x3 * y4
        + x4 * y1
        - y1 * x2
        - y2 * x3
        - y3 * x4
        - y4 * x1
    )
    area = abs(shoelace) / 2.0
    edge1 = math.hypot(x1 - x2, y1 - y2)
    edge2 = math.hypot(x2 - x3, y2 - y3)
    return area, min(edge1, edge2), max(edge1, edge2)


def summarize_annotations() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    distribution_rows: list[dict[str, object]] = []
    crane_file_rows: list[dict[str, object]] = []

    for split in ["ss_train", "ss_val"]:
        ann_dir = ROOT / "data" / "DOTA2_1024_500" / split / "annfiles"
        files = sorted(ann_dir.glob("*.txt"))
        instances: Counter[str] = Counter()
        files_with_class: defaultdict[str, set[str]] = defaultdict(set)
        areas: defaultdict[str, list[float]] = defaultdict(list)
        short_edges: defaultdict[str, list[float]] = defaultdict(list)
        long_edges: defaultdict[str, list[float]] = defaultdict(list)
        crane_by_file: defaultdict[str, Counter[str]] = defaultdict(Counter)

        for path in files:
            for raw in path.read_text(errors="replace").splitlines():
                parts = raw.split()
                if len(parts) < 9:
                    continue
                cls = parts[8]
                if cls not in CLASSES:
                    continue
                points = [float(value) for value in parts[:8]]
                difficulty = parts[9] if len(parts) > 9 else ""
                area, short_edge, long_edge = box_stats(points)
                instances[cls] += 1
                files_with_class[cls].add(path.name)
                areas[cls].append(area)
                short_edges[cls].append(short_edge)
                long_edges[cls].append(long_edge)
                if cls == "container-crane":
                    crane_by_file[path.name][difficulty] += 1

        total_instances = sum(instances.values())
        for cls in CLASSES:
            cls_areas = areas[cls]
            cls_short = short_edges[cls]
            cls_long = long_edges[cls]
            n = instances[cls]
            distribution_rows.append(
                {
                    "split": split,
                    "class": cls,
                    "instances": n,
                    "files_with_class": len(files_with_class[cls]),
                    "instance_pct": round(n / total_instances * 100, 4)
                    if total_instances
                    else 0.0,
                    "area_mean": round(sum(cls_areas) / n, 2) if n else "",
                    "area_min": round(min(cls_areas), 2) if n else "",
                    "area_max": round(max(cls_areas), 2) if n else "",
                    "short_edge_mean": round(sum(cls_short) / n, 2) if n else "",
                    "short_edge_min": round(min(cls_short), 2) if n else "",
                    "short_edge_max": round(max(cls_short), 2) if n else "",
                    "long_edge_mean": round(sum(cls_long) / n, 2) if n else "",
                    "long_edge_min": round(min(cls_long), 2) if n else "",
                    "long_edge_max": round(max(cls_long), 2) if n else "",
                }
            )

        for name, counts in sorted(crane_by_file.items()):
            crane_file_rows.append(
                {
                    "split": split,
                    "ann_file": name,
                    "instances": sum(counts.values()),
                    "difficulty_0": counts.get("0", 0),
                    "difficulty_1": counts.get("1", 0),
                    "difficulty_2": counts.get("2", 0),
                    "difficulty_blank": counts.get("", 0),
                }
            )

    return distribution_rows, crane_file_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(
    eval_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    distribution_rows: list[dict[str, object]],
) -> str:
    by_exp_epoch = {
        (row["experiment"], row["epoch"]): row for row in summary_rows
    }
    baseline4 = by_exp_epoch[("lsknet_baseline_fullinit", 4)]
    baseline12 = by_exp_epoch[("lsknet_baseline_fullinit", 12)]
    faa12 = by_exp_epoch[("faahead_lsknet_lr1e4_noema", 12)]
    train_crane = next(
        row
        for row in distribution_rows
        if row["split"] == "ss_train" and row["class"] == "container-crane"
    )
    val_crane = next(
        row
        for row in distribution_rows
        if row["split"] == "ss_val" and row["class"] == "container-crane"
    )

    def md_link(path: Path) -> str:
        return f"`{rel(path)}`"

    lines = [
        "# GPU-free DOTA2 Experiment Notes",
        "",
        "生成日期：2026-04-30",
        "",
        "## 任务顺序",
        "",
        "1. 汇总现有日志指标，固定 CSV 列顺序。",
        "2. 统计 `container-crane` 与全类别数据分布。",
        "3. 准备下一轮 GPU 实验配置。",
        "4. 对比 FAAHead 与 LSKNet baseline。",
        "5. 固化结论和后续执行顺序。",
        "",
        "## 产物",
        "",
        f"- 指标明细：`{rel(OUT_DIR / 'per_class_ap.csv')}`",
        f"- epoch 摘要：`{rel(OUT_DIR / 'epoch_summary.csv')}`",
        f"- epoch4 到 epoch12 变化：`{rel(OUT_DIR / 'class_delta_epoch4_to_12.csv')}`",
        f"- 数据分布：`{rel(OUT_DIR / 'class_distribution.csv')}`",
        f"- container-crane 切片清单：`{rel(OUT_DIR / 'container_crane_files.csv')}`",
        "",
        "## CSV 列顺序",
        "",
        "- `per_class_ap.csv`: `experiment, log_path, epoch, class, gts, dets, recall, ap, overall_map, overall_ap50, table_map, line_no`",
        "- `epoch_summary.csv`: `experiment, epoch, overall_map, overall_ap50, best_epoch_by_map, container_crane_ap, container_crane_recall, small_vehicle_ap, large_vehicle_ap, ship_ap, plane_ap`",
        "- `class_distribution.csv`: `split, class, instances, files_with_class, instance_pct, area_mean, area_min, area_max, short_edge_mean, short_edge_min, short_edge_max, long_edge_mean, long_edge_min, long_edge_max`",
        "",
        "## 关键结论",
        "",
        f"- LSKNet baseline 最好的是 epoch 4：mAP `{baseline4['overall_map']:.4f}`；epoch 12 为 `{baseline12['overall_map']:.4f}`。",
        "- 训练过程没有看到 NaN、发散或 checkpoint 失败；更像是 epoch4 后继续训练导致验证集泛化下降。",
        f"- `container-crane` 在 baseline epoch12 的 AP 为 `{baseline12['container_crane_ap']:.3f}`，recall 为 `{baseline12['container_crane_recall']:.3f}`。",
        f"- `container-crane` 数据很少：ss_train `{train_crane['instances']}` 个实例、`{train_crane['files_with_class']}` 个切片；ss_val `{val_crane['instances']}` 个实例、`{val_crane['files_with_class']}` 个切片。",
        f"- 这个类很细长：ss_train 短边均值 `{train_crane['short_edge_mean']}` px、长边均值 `{train_crane['long_edge_mean']}` px。",
        f"- FAAHead NoEMA epoch12 mAP `{faa12['overall_map']:.4f}`，低于 baseline epoch12 `{baseline12['overall_map']:.4f}`，也低于 baseline epoch4 `{baseline4['overall_map']:.4f}`；当前这组 FAAHead 不应作为主结果。",
        "",
        "## 下一轮实验顺序",
        "",
        "1. **先跑 SaveBest baseline sanity**：只加 `save_best='dota/mAP'`，不改变训练行为，用于自动保存最优 checkpoint。",
        f"   - 配置：{md_link(CONFIGS['save_best'])}",
        "2. **再跑 SaveBest + EMA**：验证 EMA 是否稳定 epoch4 后的泛化。",
        f"   - 配置：{md_link(CONFIGS['save_best_ema'])}",
        "3. **再跑 SaveBest + EarlyLR**：把 milestone 从 `[8, 11]` 改为 `[4, 8]`，对应当前 epoch4 最优的观察。",
        f"   - 配置：{md_link(CONFIGS['save_best_earlylr'])}",
        "",
        "配置解析验证：三份配置均已用 `mmengine.Config.fromfile(..., import_custom_modules=False)` 做 CPU 解析检查；",
        "`save_best='dota/mAP'` 生效，EarlyLR 的 milestone 为 `[4, 8]`，默认 hook 仍保留 logger/timer/sampler 等基础项。",
        "",
        "## 训练建议",
        "",
        "- 当前已有 LSKNet baseline 若要报告单点结果，优先使用 `work_dirs/lsknet_dotav2_ss_orcnn_bs2_fullinit/epoch_4.pth`。",
        "- 后续训练必须开启 `save_best`，否则 `last_checkpoint` 可能不是最优模型。",
        "- `container-crane` 需要单独采样或重加权实验；只改检测头或继续训练不一定能解决。",
        "- FAAHead NoEMA 当前没有超过 baseline；下一步不要优先扩大 FAAHead 搜索，先把 checkpoint 策略和 LR/EMA 控制住。",
        "",
        "## 纳入分析的日志",
        "",
    ]
    for experiment, path in LOGS.items():
        lines.append(f"- `{experiment}`: `{rel(path)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_rows: list[dict[str, object]] = []
    for experiment, path in LOGS.items():
        eval_rows.extend(parse_eval_tables(path, experiment))

    eval_fields = [
        "experiment",
        "log_path",
        "epoch",
        "class",
        "gts",
        "dets",
        "recall",
        "ap",
        "overall_map",
        "overall_ap50",
        "table_map",
        "line_no",
    ]
    write_csv(OUT_DIR / "per_class_ap.csv", eval_fields, eval_rows)

    by_exp_epoch: defaultdict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        by_exp_epoch[(str(row["experiment"]), int(row["epoch"]))].append(row)

    best_by_experiment: dict[str, int] = {}
    for experiment in LOGS:
        candidates = [
            (epoch, rows[0]["overall_map"])
            for (exp, epoch), rows in by_exp_epoch.items()
            if exp == experiment
        ]
        best_by_experiment[experiment] = max(candidates, key=lambda item: item[1])[0]

    summary_rows: list[dict[str, object]] = []
    for (experiment, epoch), rows in sorted(by_exp_epoch.items()):
        class_map = {str(row["class"]): row for row in rows}
        summary_rows.append(
            {
                "experiment": experiment,
                "epoch": epoch,
                "overall_map": rows[0]["overall_map"],
                "overall_ap50": rows[0]["overall_ap50"],
                "best_epoch_by_map": epoch == best_by_experiment[experiment],
                "container_crane_ap": class_map["container-crane"]["ap"],
                "container_crane_recall": class_map["container-crane"]["recall"],
                "small_vehicle_ap": class_map["small-vehicle"]["ap"],
                "large_vehicle_ap": class_map["large-vehicle"]["ap"],
                "ship_ap": class_map["ship"]["ap"],
                "plane_ap": class_map["plane"]["ap"],
            }
        )
    summary_fields = [
        "experiment",
        "epoch",
        "overall_map",
        "overall_ap50",
        "best_epoch_by_map",
        "container_crane_ap",
        "container_crane_recall",
        "small_vehicle_ap",
        "large_vehicle_ap",
        "ship_ap",
        "plane_ap",
    ]
    write_csv(OUT_DIR / "epoch_summary.csv", summary_fields, summary_rows)

    delta_rows: list[dict[str, object]] = []
    per_class_lookup = {
        (str(row["experiment"]), int(row["epoch"]), str(row["class"])): row
        for row in eval_rows
    }
    for cls in CLASSES:
        b4 = per_class_lookup.get(("lsknet_baseline_fullinit", 4, cls))
        b12 = per_class_lookup.get(("lsknet_baseline_fullinit", 12, cls))
        f4 = per_class_lookup.get(("faahead_lsknet_lr1e4_noema", 4, cls))
        f12 = per_class_lookup.get(("faahead_lsknet_lr1e4_noema", 12, cls))
        delta_rows.append(
            {
                "class": cls,
                "baseline_ap_e4": b4["ap"] if b4 else "",
                "baseline_ap_e12": b12["ap"] if b12 else "",
                "baseline_delta_e12_minus_e4": round(b12["ap"] - b4["ap"], 4)
                if b4 and b12
                else "",
                "faahead_ap_e4": f4["ap"] if f4 else "",
                "faahead_ap_e12": f12["ap"] if f12 else "",
                "faahead_delta_e12_minus_e4": round(f12["ap"] - f4["ap"], 4)
                if f4 and f12
                else "",
                "faahead_minus_baseline_e12": round(f12["ap"] - b12["ap"], 4)
                if f12 and b12
                else "",
            }
        )
    delta_fields = [
        "class",
        "baseline_ap_e4",
        "baseline_ap_e12",
        "baseline_delta_e12_minus_e4",
        "faahead_ap_e4",
        "faahead_ap_e12",
        "faahead_delta_e12_minus_e4",
        "faahead_minus_baseline_e12",
    ]
    write_csv(OUT_DIR / "class_delta_epoch4_to_12.csv", delta_fields, delta_rows)

    distribution_rows, crane_file_rows = summarize_annotations()
    distribution_fields = [
        "split",
        "class",
        "instances",
        "files_with_class",
        "instance_pct",
        "area_mean",
        "area_min",
        "area_max",
        "short_edge_mean",
        "short_edge_min",
        "short_edge_max",
        "long_edge_mean",
        "long_edge_min",
        "long_edge_max",
    ]
    write_csv(OUT_DIR / "class_distribution.csv", distribution_fields, distribution_rows)

    crane_fields = [
        "split",
        "ann_file",
        "instances",
        "difficulty_0",
        "difficulty_1",
        "difficulty_2",
        "difficulty_blank",
    ]
    write_csv(OUT_DIR / "container_crane_files.csv", crane_fields, crane_file_rows)

    report = make_markdown(eval_rows, summary_rows, distribution_rows)
    (OUT_DIR / "README.md").write_text(report)


if __name__ == "__main__":
    main()

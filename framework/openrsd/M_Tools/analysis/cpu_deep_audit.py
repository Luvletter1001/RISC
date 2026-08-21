#!/usr/bin/env python3
"""CPU-only audit for DOTA2 experiments, data quality, and rare classes."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "work_dirs" / "analysis_reports" / "cpu_deep_20260430"
SAMPLING_DIR = OUT_DIR / "sampling_lists"
VIS_DIR = OUT_DIR / "visualizations"

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

RARE_THRESHOLD = 500

ROW_RE = re.compile(
    r"^\|\s*(?P<class>[^|]+?)\s*\|\s*(?P<gts>\d+)\s*\|\s*"
    r"(?P<dets>\d+)\s*\|\s*(?P<recall>[0-9.]+)\s*\|\s*(?P<ap>[0-9.]+)\s*\|"
)
MAP_ROW_RE = re.compile(r"^\|\s*mAP\s*\|.*\|\s*(?P<table_map>[0-9.]+)\s*\|")
EPOCH_RE = re.compile(
    r"Epoch\(val\)\s*\[(?P<epoch>\d+)\].*dota/mAP:\s*(?P<map>[0-9.]+)"
    r"\s+dota/AP50:\s*(?P<ap50>[0-9.]+)"
)
CONFIG_RE = re.compile(r"^Config:\s*$")
EXP_NAME_RE = re.compile(r"Exp name:\s*(?P<name>\S+)")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_id(name: str) -> str:
    return name.split("__", 1)[0]


def image_for_ann(ann_path: Path, split_dir: Path) -> Path:
    return split_dir / "images" / f"{ann_path.stem}.png"


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def parse_eval_tables(log_path: Path) -> list[dict[str, object]]:
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()
    exp_name = ""
    for line in lines:
        match = EXP_NAME_RE.search(line)
        if match:
            exp_name = match.group("name")
            break

    rows: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    table_map = None
    for idx, line in enumerate(lines):
        row_match = ROW_RE.match(line)
        if row_match:
            item = row_match.groupdict()
            pending.append(
                {
                    "log_path": rel(log_path),
                    "work_dir": rel(log_path.parents[1])
                    if log_path.parent.name.startswith("20")
                    else rel(log_path.parent),
                    "exp_name": exp_name,
                    "epoch": "",
                    "class": item["class"],
                    "gts": int(item["gts"]),
                    "dets": int(item["dets"]),
                    "recall": float(item["recall"]),
                    "ap": float(item["ap"]),
                    "overall_map": "",
                    "overall_ap50": "",
                    "table_map": "",
                    "line_no": idx + 1,
                    "log_sha1_12": hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12],
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
                rows.append(item)
            pending = []
            table_map = None
    return rows


def parse_all_logs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eval_rows: list[dict[str, object]] = []
    for log_path in sorted((ROOT / "work_dirs").glob("**/*.log")):
        eval_rows.extend(parse_eval_tables(log_path))

    grouped: defaultdict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        grouped[(str(row["log_path"]), int(row["epoch"]))].append(row)

    best_rows: list[dict[str, object]] = []
    for (log_path, epoch), rows in sorted(grouped.items()):
        class_map = {str(row["class"]): row for row in rows}
        best_rows.append(
            {
                "log_path": log_path,
                "work_dir": rows[0]["work_dir"],
                "exp_name": rows[0]["exp_name"],
                "epoch": epoch,
                "overall_map": rows[0]["overall_map"],
                "overall_ap50": rows[0]["overall_ap50"],
                "container_crane_ap": class_map.get("container-crane", {}).get("ap", ""),
                "container_crane_recall": class_map.get("container-crane", {}).get("recall", ""),
                "small_vehicle_ap": class_map.get("small-vehicle", {}).get("ap", ""),
                "large_vehicle_ap": class_map.get("large-vehicle", {}).get("ap", ""),
                "ship_ap": class_map.get("ship", {}).get("ap", ""),
                "plane_ap": class_map.get("plane", {}).get("ap", ""),
                "log_sha1_12": rows[0]["log_sha1_12"],
            }
        )

    best_by_log: dict[str, float] = {}
    for row in best_rows:
        key = str(row["log_path"])
        value = float(row["overall_map"])
        if key not in best_by_log or value > best_by_log[key]:
            best_by_log[key] = value

    for row in best_rows:
        row["is_best_epoch_for_log"] = float(row["overall_map"]) == best_by_log[str(row["log_path"])]

    return eval_rows, sorted(
        best_rows, key=lambda row: (float(row["overall_map"]), str(row["log_path"])), reverse=True
    )


def scan_split(split: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, set[str]]]:
    split_dir = ROOT / "data" / "DOTA2_1024_500" / split
    ann_dir = split_dir / "annfiles"
    img_dir = split_dir / "images"
    ann_files = sorted(ann_dir.glob("*.txt"))
    image_files = sorted(img_dir.glob("*.png"))
    image_names = {path.name for path in image_files}
    ann_names = {path.with_suffix(".png").name for path in ann_files}

    image_manifest: list[dict[str, object]] = []
    class_counter: Counter[str] = Counter()
    file_counter: defaultdict[str, set[str]] = defaultdict(set)
    difficulty_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    area_values: defaultdict[str, list[float]] = defaultdict(list)
    short_values: defaultdict[str, list[float]] = defaultdict(list)
    long_values: defaultdict[str, list[float]] = defaultdict(list)
    source_by_class: defaultdict[str, set[str]] = defaultdict(set)
    file_classes: dict[str, set[str]] = {}

    empty_ann = 0
    invalid_lines = 0
    unknown_labels = Counter()
    degenerate_instances = 0
    out_of_bounds_instances = 0
    max_overflow = 0.0
    max_underflow = 0.0

    for ann_path in ann_files:
        img_path = image_for_ann(ann_path, split_dir)
        width = height = ""
        if img_path.exists():
            with Image.open(img_path) as img:
                width, height = img.size
        classes_in_file: set[str] = set()
        instance_count = 0
        raw_lines = [line for line in ann_path.read_text(errors="replace").splitlines() if line.strip()]
        if not raw_lines:
            empty_ann += 1
        for raw in raw_lines:
            parts = raw.split()
            if len(parts) < 9:
                invalid_lines += 1
                continue
            cls = parts[8]
            if cls not in CLASSES:
                unknown_labels[cls] += 1
                continue
            points = [float(value) for value in parts[:8]]
            difficulty = parts[9] if len(parts) > 9 else ""
            area, short_edge, long_edge = box_stats(points)
            if area <= 1.0 or short_edge <= 1.0:
                degenerate_instances += 1
            if width and height:
                xs = points[0::2]
                ys = points[1::2]
                overflow = max(max(xs) - width, max(ys) - height, 0)
                underflow = max(-min(xs), -min(ys), 0)
                if overflow > 0 or underflow > 0:
                    out_of_bounds_instances += 1
                max_overflow = max(max_overflow, overflow)
                max_underflow = max(max_underflow, underflow)
            class_counter[cls] += 1
            file_counter[cls].add(ann_path.name)
            difficulty_counter[cls][difficulty] += 1
            area_values[cls].append(area)
            short_values[cls].append(short_edge)
            long_values[cls].append(long_edge)
            source_by_class[cls].add(source_id(ann_path.name))
            classes_in_file.add(cls)
            instance_count += 1
        file_classes[ann_path.name] = classes_in_file
        image_manifest.append(
            {
                "split": split,
                "image_file": img_path.name,
                "ann_file": ann_path.name,
                "source_id": source_id(ann_path.name),
                "width": width,
                "height": height,
                "has_image": img_path.name in image_names,
                "instances": instance_count,
                "classes": " ".join(sorted(classes_in_file)),
            }
        )

    total_instances = sum(class_counter.values())
    class_rows: list[dict[str, object]] = []
    for cls in CLASSES:
        n = class_counter[cls]
        class_rows.append(
            {
                "split": split,
                "class": cls,
                "instances": n,
                "files_with_class": len(file_counter[cls]),
                "sources_with_class": len(source_by_class[cls]),
                "instance_pct": round(n / total_instances * 100, 4) if total_instances else 0,
                "difficulty_0": difficulty_counter[cls].get("0", 0),
                "difficulty_1": difficulty_counter[cls].get("1", 0),
                "difficulty_2": difficulty_counter[cls].get("2", 0),
                "difficulty_blank": difficulty_counter[cls].get("", 0),
                "area_mean": round(sum(area_values[cls]) / n, 2) if n else "",
                "area_min": round(min(area_values[cls]), 2) if n else "",
                "area_max": round(max(area_values[cls]), 2) if n else "",
                "short_edge_mean": round(sum(short_values[cls]) / n, 2) if n else "",
                "long_edge_mean": round(sum(long_values[cls]) / n, 2) if n else "",
            }
        )

    audit_rows = [
        {
            "split": split,
            "ann_files": len(ann_files),
            "image_files": len(image_files),
            "images_missing_ann": len(image_names - ann_names),
            "anns_missing_image": len(ann_names - image_names),
            "empty_ann_files": empty_ann,
            "invalid_lines": invalid_lines,
            "unknown_label_instances": sum(unknown_labels.values()),
            "unknown_labels": " ".join(f"{k}:{v}" for k, v in sorted(unknown_labels.items())),
            "degenerate_instances": degenerate_instances,
            "out_of_bounds_instances": out_of_bounds_instances,
            "max_overflow_px": round(max_overflow, 2),
            "max_underflow_px": round(max_underflow, 2),
            "total_instances": total_instances,
        }
    ]

    co_rows: list[dict[str, object]] = []
    for a in CLASSES:
        for b in CLASSES:
            if a <= b:
                count = sum(1 for classes in file_classes.values() if a in classes and b in classes)
                co_rows.append({"split": split, "class_a": a, "class_b": b, "files_with_both": count})

    split_sets = {
        "source_ids": {source_id(path.name) for path in ann_files},
        "image_names": image_names,
        "ann_names": {path.name for path in ann_files},
    }
    return image_manifest, class_rows + audit_rows, co_rows, split_sets


def build_dataset_audit() -> dict[str, object]:
    image_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    co_rows: list[dict[str, object]] = []
    split_sets: dict[str, dict[str, set[str]]] = {}

    for split in ["ss_train", "ss_val", "rot_val_standard"]:
        split_dir = ROOT / "data" / "DOTA2_1024_500" / split
        if not split_dir.exists():
            continue
        images, mixed_rows, co, sets = scan_split(split)
        image_rows.extend(images)
        for row in mixed_rows:
            if "class" in row:
                class_rows.append(row)
            else:
                audit_rows.append(row)
        co_rows.extend(co)
        split_sets[split] = sets

    overlap_rows: list[dict[str, object]] = []
    for left, right in [("ss_train", "ss_val"), ("ss_train", "rot_val_standard"), ("ss_val", "rot_val_standard")]:
        if left in split_sets and right in split_sets:
            overlap = sorted(split_sets[left]["source_ids"] & split_sets[right]["source_ids"])
            overlap_rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "overlap_source_count": len(overlap),
                    "overlap_sources": " ".join(overlap[:200]),
                }
            )

    return {
        "image_rows": image_rows,
        "class_rows": class_rows,
        "audit_rows": audit_rows,
        "co_rows": co_rows,
        "overlap_rows": overlap_rows,
    }


def build_sampling_lists(class_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    train_counts = {
        str(row["class"]): int(row["instances"])
        for row in class_rows
        if row["split"] == "ss_train"
    }
    max_count = max(train_counts.values())
    suggestions: list[dict[str, object]] = []
    for cls in CLASSES:
        count = train_counts[cls]
        repeat = min(10, max(1, math.ceil(math.sqrt(max_count / max(count, 1)))))
        suggestions.append(
            {
                "class": cls,
                "train_instances": count,
                "repeat_factor_suggestion": repeat,
                "rare_under_500": count < RARE_THRESHOLD,
            }
        )

    ann_dir = ROOT / "data" / "DOTA2_1024_500" / "ss_train" / "annfiles"
    img_dir = ROOT / "data" / "DOTA2_1024_500" / "ss_train" / "images"
    rare_classes = {row["class"] for row in suggestions if row["rare_under_500"]}
    by_class_paths: defaultdict[str, set[str]] = defaultdict(set)
    rare_paths: set[str] = set()
    crane_paths: set[str] = set()
    for ann_path in sorted(ann_dir.glob("*.txt")):
        classes = set()
        for raw in ann_path.read_text(errors="replace").splitlines():
            parts = raw.split()
            if len(parts) >= 9 and parts[8] in CLASSES:
                classes.add(parts[8])
        image_path = rel(img_dir / f"{ann_path.stem}.png")
        for cls in classes:
            by_class_paths[cls].add(image_path)
        if classes & rare_classes:
            rare_paths.add(image_path)
        if "container-crane" in classes:
            crane_paths.add(image_path)

    SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLING_DIR / "container_crane_train_tiles.txt").write_text(
        "\n".join(sorted(crane_paths)) + "\n"
    )
    (SAMPLING_DIR / "rare_under_500_train_tiles.txt").write_text(
        "\n".join(sorted(rare_paths)) + "\n"
    )
    for cls, paths in by_class_paths.items():
        if cls in rare_classes:
            safe = cls.replace("/", "_")
            (SAMPLING_DIR / f"{safe}_train_tiles.txt").write_text("\n".join(sorted(paths)) + "\n")
    return suggestions


def ann_records_for_class(split: str, target_class: str) -> list[dict[str, object]]:
    split_dir = ROOT / "data" / "DOTA2_1024_500" / split
    rows: list[dict[str, object]] = []
    for ann_path in sorted((split_dir / "annfiles").glob("*.txt")):
        boxes = []
        for raw in ann_path.read_text(errors="replace").splitlines():
            parts = raw.split()
            if len(parts) >= 9 and parts[8] == target_class:
                boxes.append(
                    {
                        "points": [float(value) for value in parts[:8]],
                        "difficulty": parts[9] if len(parts) > 9 else "",
                    }
                )
        if boxes:
            rows.append(
                {
                    "split": split,
                    "ann_file": ann_path.name,
                    "image_path": split_dir / "images" / f"{ann_path.stem}.png",
                    "boxes": boxes,
                }
            )
    return rows


def draw_contact_sheets(records: list[dict[str, object]], split: str, per_page: int = 25) -> list[Path]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    thumb_w = 240
    thumb_h = 240
    label_h = 34
    cols = 5
    rows_per_page = math.ceil(per_page / cols)
    font = ImageFont.load_default()

    for page, start in enumerate(range(0, len(records), per_page), start=1):
        subset = records[start : start + per_page]
        sheet = Image.new("RGB", (cols * thumb_w, rows_per_page * (thumb_h + label_h)), "white")
        draw_sheet = ImageDraw.Draw(sheet)
        for idx, record in enumerate(subset):
            row = idx // cols
            col = idx % cols
            x0 = col * thumb_w
            y0 = row * (thumb_h + label_h)
            image_path = Path(record["image_path"])
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.new("RGB", (1024, 1024), "gray")
            orig_w, orig_h = image.size
            scale = min(thumb_w / orig_w, thumb_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            image = image.resize((new_w, new_h))
            canvas = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
            off_x = (thumb_w - new_w) // 2
            off_y = (thumb_h - new_h) // 2
            canvas.paste(image, (off_x, off_y))
            draw = ImageDraw.Draw(canvas)
            for box in record["boxes"]:
                points = box["points"]
                scaled = [
                    (points[i] * scale + off_x, points[i + 1] * scale + off_y)
                    for i in range(0, 8, 2)
                ]
                color = {"0": "red", "1": "orange", "2": "purple"}.get(
                    str(box["difficulty"]), "blue"
                )
                draw.line(scaled + [scaled[0]], fill=color, width=2)
            sheet.paste(canvas, (x0, y0))
            title = f"{record['ann_file']} n={len(record['boxes'])}"
            draw_sheet.text((x0 + 3, y0 + thumb_h + 3), title[:38], fill="black", font=font)
        output = VIS_DIR / f"container_crane_{split}_contact_sheet_p{page}.jpg"
        sheet.save(output, quality=90)
        outputs.append(output)
    return outputs


def build_visualizations() -> list[Path]:
    outputs: list[Path] = []
    for split in ["ss_train", "ss_val"]:
        records = ann_records_for_class(split, "container-crane")
        outputs.extend(draw_contact_sheets(records, split))

    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>container-crane contact sheets</title>",
        "<h1>container-crane contact sheets</h1>",
    ]
    for path in outputs:
        html_lines.append(f"<h2>{path.name}</h2>")
        html_lines.append(f"<img src='{path.name}' style='max-width:100%;height:auto'>")
    html = VIS_DIR / "container_crane_contact_sheets.html"
    html.write_text("\n".join(html_lines) + "\n")
    outputs.append(html)
    return outputs


def make_report(
    best_epoch_rows: list[dict[str, object]],
    dataset: dict[str, object],
    suggestions: list[dict[str, object]],
    visual_outputs: list[Path],
) -> None:
    ranked = [row for row in best_epoch_rows if row["is_best_epoch_for_log"]]
    top_ranked = ranked[:15]
    audit_rows = dataset["audit_rows"]
    overlap_rows = dataset["overlap_rows"]
    class_rows = dataset["class_rows"]
    train_crane = next(
        row for row in class_rows if row["split"] == "ss_train" and row["class"] == "container-crane"
    )
    val_crane = next(
        row for row in class_rows if row["split"] == "ss_val" and row["class"] == "container-crane"
    )
    rare = [row for row in suggestions if row["rare_under_500"]]

    lines = [
        "# CPU Deep Audit",
        "",
        "生成日期：2026-04-30",
        "",
        "## 本次用 CPU 完成的任务",
        "",
        "1. 全量扫描 `work_dirs` 下 63 个日志，解析所有 DOTA mAP 表。",
        "2. 生成日志排行榜、逐类指标、每个日志最优 epoch。",
        "3. 审计 `ss_train`、`ss_val`、`rot_val_standard` 的标注、图片、越界框、空标注和源图重叠。",
        "4. 为长尾类生成训练切片清单和 repeat-factor 建议。",
        "5. 生成 `container-crane` 标注 contact sheet，便于人工检查标注质量。",
        "",
        "## 主要产物",
        "",
        f"- 全量逐类指标：`{rel(OUT_DIR / 'all_per_class_eval.csv')}`",
        f"- 日志/epoch 排行榜：`{rel(OUT_DIR / 'ranked_log_epochs.csv')}`",
        f"- 数据集审计：`{rel(OUT_DIR / 'dataset_audit.csv')}`",
        f"- 类别分布增强版：`{rel(OUT_DIR / 'class_distribution_detailed.csv')}`",
        f"- 图片 manifest：`{rel(OUT_DIR / 'image_manifest.csv')}`",
        f"- 源图重叠检查：`{rel(OUT_DIR / 'source_overlap.csv')}`",
        f"- 类共现矩阵：`{rel(OUT_DIR / 'class_cooccurrence.csv')}`",
        f"- 采样建议：`{rel(SAMPLING_DIR / 'repeat_factor_suggestions.csv')}`",
        f"- container-crane 训练切片：`{rel(SAMPLING_DIR / 'container_crane_train_tiles.txt')}`",
        f"- 长尾类训练切片：`{rel(SAMPLING_DIR / 'rare_under_500_train_tiles.txt')}`",
        f"- container-crane 可视化入口：`{rel(VIS_DIR / 'container_crane_contact_sheets.html')}`",
        f"- ClassBalanced 配置：`M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest_ClassBalanced.py`",
        f"- EMA + ClassBalanced 配置：`M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest_EMA_ClassBalanced.py`",
        "",
        "## 日志排行榜 Top 15",
        "",
        "| rank | mAP | epoch | container-crane AP | log |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(top_ranked, start=1):
        lines.append(
            f"| {idx} | {float(row['overall_map']):.4f} | {row['epoch']} | "
            f"{row['container_crane_ap']} | `{row['log_path']}` |"
        )

    lines.extend(
        [
            "",
            "## 数据审计摘要",
            "",
            "| split | ann_files | image_files | missing_ann | missing_image | empty_ann | out_of_bounds | max_overflow_px | total_instances |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in audit_rows:
        lines.append(
            f"| {row['split']} | {row['ann_files']} | {row['image_files']} | "
            f"{row['images_missing_ann']} | {row['anns_missing_image']} | {row['empty_ann_files']} | "
            f"{row['out_of_bounds_instances']} | {row['max_overflow_px']} | {row['total_instances']} |"
        )
    lines.extend(
        [
            "",
            "说明：越界框数量较高，主要来自切片后目标跨 tile 边界；当前训练使用 `DOTADatasetClamp`，",
            "因此这不是立即判定为坏数据，但仍建议在可视化中重点检查截断目标。",
        ]
    )

    lines.extend(
        [
            "",
            "## 源图重叠",
            "",
            "| left | right | overlap source count | notes |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in overlap_rows:
        note = "需要确认 split 生成策略" if int(row["overlap_source_count"]) else "无源图重叠"
        lines.append(
            f"| {row['left_split']} | {row['right_split']} | {row['overlap_source_count']} | {note} |"
        )

    lines.extend(
        [
            "",
            "## container-crane",
            "",
            f"- ss_train: `{train_crane['instances']}` 个实例，`{train_crane['files_with_class']}` 个切片，`{train_crane['sources_with_class']}` 个源图。",
            f"- ss_val: `{val_crane['instances']}` 个实例，`{val_crane['files_with_class']}` 个切片，`{val_crane['sources_with_class']}` 个源图。",
            f"- ss_train 平均短边 `{train_crane['short_edge_mean']}` px，平均长边 `{train_crane['long_edge_mean']}` px。",
            "- contact sheet 中红色为 difficulty 0，橙色为 1，紫色为 2。",
            "",
            "## 长尾类采样建议",
            "",
            "| class | train instances | repeat factor suggestion |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rare:
        lines.append(
            f"| {row['class']} | {row['train_instances']} | {row['repeat_factor_suggestion']} |"
        )

    lines.extend(
        [
            "",
            "## 后续 CPU 任务队列",
            "",
            "1. 人工检查 contact sheet，记录疑似错标、截断严重和极细长目标。",
            "2. 根据 `source_overlap.csv` 判断是否需要重新 split，尤其关注 ss_train/ss_val 的源图重叠。",
            "3. 用 `rare_under_500_train_tiles.txt` 准备 class-balanced sampler 实验。",
            "4. GPU 恢复后优先跑 `SaveBest`、`SaveBest_EMA`、`SaveBest_EarlyLR` 三组，而不是继续扩大 FAAHead 搜索。",
            "5. 若前三组仍不能恢复 `container-crane`，再跑 `SaveBest_ClassBalanced` 和 `SaveBest_EMA_ClassBalanced`。",
            "",
            "## 可视化文件",
            "",
        ]
    )
    for path in visual_outputs:
        lines.append(f"- `{rel(path)}`")

    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    eval_rows, ranked_rows = parse_all_logs()
    eval_fields = [
        "log_path",
        "work_dir",
        "exp_name",
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
        "log_sha1_12",
    ]
    write_csv(OUT_DIR / "all_per_class_eval.csv", eval_fields, eval_rows)

    ranked_fields = [
        "log_path",
        "work_dir",
        "exp_name",
        "epoch",
        "overall_map",
        "overall_ap50",
        "container_crane_ap",
        "container_crane_recall",
        "small_vehicle_ap",
        "large_vehicle_ap",
        "ship_ap",
        "plane_ap",
        "log_sha1_12",
        "is_best_epoch_for_log",
    ]
    write_csv(OUT_DIR / "ranked_log_epochs.csv", ranked_fields, ranked_rows)

    dataset = build_dataset_audit()
    write_csv(
        OUT_DIR / "image_manifest.csv",
        ["split", "image_file", "ann_file", "source_id", "width", "height", "has_image", "instances", "classes"],
        dataset["image_rows"],
    )
    write_csv(
        OUT_DIR / "class_distribution_detailed.csv",
        [
            "split",
            "class",
            "instances",
            "files_with_class",
            "sources_with_class",
            "instance_pct",
            "difficulty_0",
            "difficulty_1",
            "difficulty_2",
            "difficulty_blank",
            "area_mean",
            "area_min",
            "area_max",
            "short_edge_mean",
            "long_edge_mean",
        ],
        dataset["class_rows"],
    )
    write_csv(
        OUT_DIR / "dataset_audit.csv",
        [
            "split",
            "ann_files",
            "image_files",
            "images_missing_ann",
            "anns_missing_image",
            "empty_ann_files",
            "invalid_lines",
            "unknown_label_instances",
            "unknown_labels",
            "degenerate_instances",
            "out_of_bounds_instances",
            "max_overflow_px",
            "max_underflow_px",
            "total_instances",
        ],
        dataset["audit_rows"],
    )
    write_csv(
        OUT_DIR / "source_overlap.csv",
        ["left_split", "right_split", "overlap_source_count", "overlap_sources"],
        dataset["overlap_rows"],
    )
    write_csv(
        OUT_DIR / "class_cooccurrence.csv",
        ["split", "class_a", "class_b", "files_with_both"],
        dataset["co_rows"],
    )

    suggestions = build_sampling_lists(dataset["class_rows"])
    write_csv(
        SAMPLING_DIR / "repeat_factor_suggestions.csv",
        ["class", "train_instances", "repeat_factor_suggestion", "rare_under_500"],
        suggestions,
    )

    visual_outputs = build_visualizations()
    make_report(ranked_rows, dataset, suggestions, visual_outputs)


if __name__ == "__main__":
    main()

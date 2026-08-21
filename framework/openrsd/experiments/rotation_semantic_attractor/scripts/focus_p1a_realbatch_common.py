"""Shared helpers for FOCUS P1A real-batch unblock scripts."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXP_SUBDIRS = [
    "preflight",
    "batch_match",
    "target_injection",
    "one_batch_real",
    "debug_overlays",
    "reports",
    "tables",
    "figures",
    "logs",
    "manifests",
    "configs",
]
DOTA2_CLASSES = [
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


def resolve(repo_root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def ensure_exp_tree(exp_dir: Path) -> None:
    for subdir in EXP_SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def safe_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]],
              fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def bool_from(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def html_from_markdown(md_path: Path, html_path: Path,
                       title: str = "FOCUS P1A Realbatch Report") -> None:
    import html

    body = html.escape(md_path.read_text(encoding="utf-8"))
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1160px;"
        "margin:32px auto;line-height:1.5}pre{white-space:pre-wrap}"
        "code{background:#f3f4f6;padding:1px 4px;border-radius:3px}"
        "</style></head><body><pre>"
        + body + "</pre></body></html>",
        encoding="utf-8")


def valid_focus_targets(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if bool_from(row.get("valid_for_loss"))
        and row.get("target_role") in {"anti_negative", "preserve_positive"}
    ]


def target_role_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row.get("target_role", "") for row in rows))


def ann_path_from_image(image_path: str | Path) -> Path:
    text = str(image_path)
    text = text.replace("/images/", "/annfiles/")
    return Path(text).with_suffix(".txt")


def image_angle_dir(angle: Any) -> str:
    try:
        return f"angle_{int(float(angle)):03d}"
    except (TypeError, ValueError):
        return "angle_UNKNOWN"


def parse_dota_ann(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            qbox = [float(v) for v in parts[:8]]
        except ValueError:
            continue
        objects.append({
            "qbox": qbox,
            "class_name": parts[8],
            "difficulty": parts[9] if len(parts) > 9 else "0",
        })
    return objects


def polygon_area(poly: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def qbox_to_cxcywha(qbox: list[float]) -> list[float]:
    pts = [(qbox[i], qbox[i + 1]) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    edge01 = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    edge12 = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    width = max(edge01, edge12, 1.0)
    height = max(min(edge01, edge12), 1.0)
    angle = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    return [cx, cy, width, height, angle]


def group_targets_by_image(
        rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("image_path", "")].append(row)
    return dict(grouped)


def summarize_selected_image(
        image_path: str,
        rows: list[dict[str, str]]) -> dict[str, Any]:
    roles = Counter(row.get("target_role") for row in rows)
    first = rows[0] if rows else {}
    ann_path = ann_path_from_image(image_path)
    return {
        "image_path": image_path,
        "annotation_path": str(ann_path),
        "tile_id": first.get("tile_id", Path(image_path).stem),
        "angle": int(float(first.get("angle", 0) or 0)),
        "coordinate_frame": first.get("coordinate_frame", ""),
        "anti_target_count": int(roles.get("anti_negative", 0)),
        "preserve_target_count": int(roles.get("preserve_positive", 0)),
        "image_exists": Path(image_path).exists(),
        "annotation_exists": ann_path.exists(),
        "annotation_object_count": len(parse_dota_ann(ann_path)),
        "target_ids": [row.get("focus_target_id", "") for row in rows],
        "targets": rows,
    }


def select_realbatch_images(target_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = group_targets_by_image(target_rows)
    eligible = [
        (image_path, rows) for image_path, rows in grouped.items()
        if image_path and Path(image_path).exists()
        and ann_path_from_image(image_path).exists()
    ]
    for image_path, rows in eligible:
        roles = Counter(row.get("target_role") for row in rows)
        if roles.get("anti_negative", 0) > 0 and roles.get("preserve_positive", 0) > 0:
            return [summarize_selected_image(image_path, rows)]

    anti = next(
        ((image_path, rows) for image_path, rows in eligible
         if any(row.get("target_role") == "anti_negative" for row in rows)),
        None)
    preserve = next(
        ((image_path, rows) for image_path, rows in eligible
         if any(row.get("target_role") == "preserve_positive" for row in rows)),
        None)
    selected: list[dict[str, Any]] = []
    if anti:
        image_path, rows = anti
        selected.append(summarize_selected_image(
            image_path,
            [row for row in rows if row.get("target_role") == "anti_negative"]))
    if preserve:
        image_path, rows = preserve
        if not selected or selected[0]["image_path"] != image_path:
            selected.append(summarize_selected_image(
                image_path,
                [row for row in rows
                 if row.get("target_role") == "preserve_positive"]))
    return selected


def write_manifest(exp_dir: Path, repo_root: Path,
                   extra: dict[str, Any] | None = None) -> None:
    payload = {
        "experiment": "focus_ovd_p1a_realbatch_unblock",
        "repo_root": str(repo_root),
        "git_commit": safe_git_commit(repo_root),
        "subdirs": EXP_SUBDIRS,
    }
    if extra:
        payload.update(extra)
    write_json(exp_dir / "manifest.json", payload)
    write_json(exp_dir / "manifests/manifest.json", payload)

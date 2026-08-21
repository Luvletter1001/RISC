#!/usr/bin/env python3
"""Merge experiment Markdown records into one deduplicated summary.

This script is intentionally self-contained: it only reads Markdown files,
parses common experiment metadata/tables/paths, and writes a new summary plus
an optional JSON deduplication report. It never modifies source Markdown files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()

MASTER_COLUMNS = [
    "dataset",
    "granularity",
    "model",
    "setting",
    "split",
    "rotation",
    "angle",
    "prompt_set",
    "head",
    "status",
    "mAP",
    "AP50",
    "AP07",
    "AP12",
    "recall",
    "dets/img",
    "range",
    "RSI",
    "checkpoint",
    "config",
    "predictions",
    "log",
    "source_file",
]

ROW_KEY_FIELDS = [
    "dataset",
    "granularity",
    "model",
    "setting",
    "split",
    "rotation",
    "angle",
    "prompt_set",
    "head",
    "checkpoint",
    "config",
]

EXPERIMENT_KEY_FIELDS = [
    "dataset",
    "model",
    "config",
    "checkpoint",
    "split",
    "angle",
    "rotation",
    "setting",
    "prompt_set",
    "head",
    "tta setting",
    "out_root",
    "work_dir",
]

METRIC_FIELDS = [
    "mAP",
    "AP50",
    "AP07",
    "AP12",
    "mean",
    "min",
    "max",
    "range",
    "RSI",
    "recall",
    "recall@50",
    "dets/img",
    "correct",
    "wrong class",
    "loc fail",
    "missed",
    "false pos",
]

STATUS_OK = {"ok", "done", "success", "completed", "complete", "pass", "passed"}

MODEL_PATTERNS = [
    (re.compile(r"rtmdet[-_\s]*l", re.I), "RTMDet-L"),
    (re.compile(r"h2rbox[-_\s]*v?2", re.I), "H2RBox-v2"),
    (re.compile(r"h2rbox", re.I), "H2RBox"),
    (re.compile(r"retinanet", re.I), "Rotated RetinaNet"),
    (re.compile(r"redet", re.I), "ReDet"),
    (re.compile(r"openrsd|ovd|open[-_\s]*vocab", re.I), "OpenRSD OVD"),
]

DATASET_PATTERNS = [
    (re.compile(r"dota\s*1|dota1|dotav1|dota_?v1", re.I), "DOTA1"),
    (re.compile(r"dota\s*2|dota2", re.I), "DOTA2"),
    (re.compile(r"far\s*1m|far1m", re.I), "FAR1M"),
]

HEADER_ALIASES = {
    "dataset": "dataset",
    "data": "dataset",
    "data path": "dataset",
    "data_path": "dataset",
    "model": "model",
    "detector": "model",
    "method": "model",
    "granularity": "granularity",
    "setting": "setting",
    "strategy": "setting",
    "split": "split",
    "rotation": "rotation",
    "target angle": "angle",
    "angle": "angle",
    "source angle": "rotation",
    "prompt set": "prompt_set",
    "prompt_set": "prompt_set",
    "prompt family": "prompt_set",
    "prompt_family": "prompt_set",
    "head": "head",
    "head type": "head",
    "status": "status",
    "map": "mAP",
    "mAP": "mAP",
    "dota/map": "mAP",
    "ap50": "AP50",
    "dota/ap50": "AP50",
    "mean ap50": "AP50",
    "ap07": "AP07",
    "ap12": "AP12",
    "recall": "recall",
    "recall@50": "recall@50",
    "matched recall@0.5": "recall@50",
    "det/img": "dets/img",
    "dets/img": "dets/img",
    "detections/image": "dets/img",
    "range": "range",
    "rsi": "RSI",
    "checkpoint": "checkpoint",
    "ckpt": "checkpoint",
    "config": "config",
    "prediction": "predictions",
    "predictions": "predictions",
    "predictions pkl": "predictions",
    "merged predictions": "predictions",
    "log": "log",
    "stdout": "log",
    "stderr": "log",
    "single log": "log",
    "tta log": "log",
}

PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:/data1|/data|/home|work_dirs|resultmd|results|outputs|data|M_configs|M_Tools|tools)"
    r"[^\s`|)>\]}\"']+"
    r")"
)


@dataclass
class ParsedTable:
    index: int
    start_line: int
    heading: str
    headers: list[str]
    rows: list[dict[str, str]]
    warnings: list[str] = field(default_factory=list)


@dataclass
class SourceFile:
    path: Path
    relative_path: str
    filename: str
    mtime: str
    mtime_ts: float
    size: int
    sha256: str
    first_heading: str
    text: str
    metadata: dict[str, str]
    generated_at: str
    out_root: str
    result_csv: str
    tables: list[ParsedTable]
    important_paths: dict[str, list[str]]
    rows: list[dict[str, Any]]
    warnings: list[str]
    is_summary: bool


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def now_text() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_for_backup() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_space(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", " ; ", text, flags=re.I)
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(value: str) -> str:
    text = normalize_space(value).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[/\\]+", "/", text)
    text = re.sub(r"[^a-z0-9@.+/ -]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def md_escape(value: Any) -> str:
    text = "NA" if value is None or value == "" else str(value)
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text


def strip_outer_pipes(line: str) -> str:
    line = line.rstrip("\n")
    if line.strip().startswith("|"):
        line = line.strip()[1:]
    if line.strip().endswith("|"):
        line = line.strip()[:-1]
    return line


def split_md_row(line: str) -> list[str]:
    return [normalize_space(cell) for cell in strip_outer_pipes(line).split("|")]


def is_separator_row(line: str) -> bool:
    cells = split_md_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r"\s*:?-{3,}:?\s*", c or "") for c in cells)


def heading_level(line: str) -> tuple[int, str] | None:
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not m:
        return None
    return len(m.group(1)), normalize_space(m.group(2))


def extract_first_heading(text: str) -> str:
    for line in text.splitlines():
        h = heading_level(line)
        if h and h[0] <= 3:
            return h[1]
    return "NA"


def parse_bullet_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r"^\s*[-*]\s*([^:]{2,80}):\s*(.+?)\s*$", line)
        if not m:
            continue
        key = normalize_key(m.group(1))
        val = normalize_space(m.group(2))
        val = val.strip("` ")
        metadata.setdefault(key, val)
    return metadata


def parse_generated_at(text: str, metadata: dict[str, str]) -> str:
    for key in ("generated at", "generated_at", "time", "run ts", "run_ts"):
        norm = normalize_key(key)
        if norm in metadata:
            return metadata[norm]
    m = re.search(r"generated[_ ]at[:：]\s*`?([^`\n]+)`?", text, re.I)
    return normalize_space(m.group(1)) if m else "NA"


def metadata_value(metadata: dict[str, str], *keys: str) -> str:
    for key in keys:
        norm = normalize_key(key)
        if norm in metadata:
            return metadata[norm]
    return "NA"


def parse_tables(text: str) -> tuple[list[ParsedTable], list[str]]:
    lines = text.splitlines()
    warnings: list[str] = []
    tables: list[ParsedTable] = []
    in_code = False
    current_headings: dict[int, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        h = heading_level(line)
        if h:
            level, title = h
            current_headings[level] = title
            for k in list(current_headings):
                if k > level:
                    current_headings.pop(k, None)
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and is_separator_row(lines[i + 1]):
            headers = split_md_row(line)
            start = i + 1
            rows: list[dict[str, str]] = []
            j = i + 2
            local_warnings: list[str] = []
            while j < len(lines) and "|" in lines[j] and not lines[j].strip().startswith("```"):
                if not lines[j].strip():
                    break
                cells = split_md_row(lines[j])
                if len(cells) != len(headers):
                    local_warnings.append(
                        f"table {len(tables)+1} line {j+1}: column count {len(cells)} != header count {len(headers)}"
                    )
                if len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                elif len(cells) > len(headers):
                    cells = cells[: len(headers) - 1] + [" | ".join(cells[len(headers) - 1 :])]
                row = {headers[k] or f"col_{k+1}": cells[k] for k in range(len(headers))}
                if any(v != "" for v in row.values()):
                    rows.append(row)
                j += 1
            heading = " / ".join(current_headings[k] for k in sorted(current_headings) if k <= 3)
            tables.append(
                ParsedTable(
                    index=len(tables) + 1,
                    start_line=start,
                    heading=heading or "NA",
                    headers=headers,
                    rows=rows,
                    warnings=local_warnings,
                )
            )
            warnings.extend(local_warnings)
            i = j
            continue
        i += 1
    return tables, warnings


def canonical_header(header: str) -> str:
    norm = normalize_key(header)
    if "single" in norm and "ap50" in norm:
        return "single AP50"
    if ("tta" in norm or "merged" in norm) and "ap50" in norm:
        return "TTA AP50"
    if "default" in norm and "tta" in norm and "ap50" in norm:
        return "TTA AP50"
    if "single" in norm and norm.endswith("mean"):
        return "single mean"
    if "tta" in norm and norm.endswith("mean"):
        return "TTA mean"
    return HEADER_ALIASES.get(norm, HEADER_ALIASES.get(norm.replace(" ", "_"), header))


def parse_float(value: Any) -> float | None:
    text = normalize_space(value)
    if not text or text.lower() in {"na", "n/a", "nan", "none", "-", "--"}:
        return None
    m = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def fmt_num(value: Any) -> str:
    val = parse_float(value)
    if val is None:
        return "NA"
    return f"{val:.4f}"


def metric_signature(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sig = []
    for field_name in METRIC_FIELDS:
        if field_name in row and normalize_space(row.get(field_name, "")) not in {"", "NA"}:
            val = parse_float(row[field_name])
            sig.append((field_name, f"{val:.6g}" if val is not None else normalize_space(row[field_name])))
    return tuple(sig)


def infer_dataset(*texts: str) -> str:
    joined = " ".join(t for t in texts if t)
    for pattern, value in DATASET_PATTERNS:
        if pattern.search(joined):
            return value
    return "NA"


def infer_model(*texts: str) -> str:
    joined = " ".join(t for t in texts if t)
    for pattern, value in MODEL_PATTERNS:
        if pattern.search(joined):
            return value
    return "NA"


def infer_angle(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return "NA"
    m = re.search(r"(?:angle[_ -]?)?(\d{1,3})(?:\s*°)?", text, re.I)
    if not m:
        return text
    try:
        n = int(m.group(1))
    except ValueError:
        return text
    if 0 <= n <= 360:
        return f"{n:03d}"
    return text


def clean_status(value: Any) -> str:
    text = normalize_space(value)
    return text if text else "NA"


def classify_path(path: str) -> list[str]:
    clean = path.strip().strip("`.,;")
    lower = clean.lower()
    cats: list[str] = []
    if "predictions.pkl" in lower or "merged_predictions.pkl" in lower:
        cats.append("prediction_files")
    if lower.endswith(".log") or "stdout.log" in lower or "stderr.log" in lower or "train.log" in lower or "test.log" in lower:
        cats.append("log_files")
    if lower.endswith(".pth"):
        cats.append("checkpoints")
    if lower.endswith(".csv") or lower.endswith(".tsv"):
        cats.append("csv_tsv_result_files")
    if lower.endswith(".py") and ("config" in lower or "m_configs" in lower or "/configs/" in lower):
        cats.append("configs")
    if "work_dirs/" in lower or "/work_dirs/" in lower:
        cats.append("work_dirs")
    if lower.startswith("data/") or "/data/" in lower or "/data1/" in lower:
        cats.append("data_paths")
    return cats


def extract_paths(text: str) -> dict[str, list[str]]:
    paths: dict[str, set[str]] = defaultdict(set)
    for m in PATH_RE.finditer(text):
        path = m.group("path").strip().strip("`.,;")
        path = re.sub(r"[)>}\]]+$", "", path)
        if not path:
            continue
        for cat in classify_path(path):
            paths[cat].add(path)
    return {k: sorted(v) for k, v in paths.items()}


def get_git_hash(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return "NA"
    value = proc.stdout.strip()
    return value or "NA"


def context_parts(table_heading: str) -> list[str]:
    return [p.strip() for p in table_heading.split("/") if p.strip()]


def infer_prompt_from_heading(heading: str) -> str:
    parts = context_parts(heading)
    if not parts:
        return "NA"
    last = parts[-1]
    if infer_model(last) != "NA":
        return "NA"
    if re.search(r"prompt|raw|noun|orientation|shape|context|family|class_names", last, re.I):
        return last
    return "NA"


def infer_setting_from_heading(heading: str) -> str:
    parts = context_parts(heading)
    if not parts:
        return "NA"
    last = parts[-1]
    if re.search(r"single|tta|ensemble|fusion|alignment|baseline|consistency|ablation", last, re.I):
        return last
    return "NA"


def normalize_row_from_table(
    source: SourceFile | None,
    table: ParsedTable,
    raw_row: dict[str, str],
    source_rel: str,
    filename: str,
) -> list[dict[str, Any]]:
    metadata = source.metadata if source else {}
    file_context = " ".join([filename, table.heading, source.first_heading if source else ""])
    base: dict[str, Any] = {"source_file": source_rel, "source_table": table.index}
    raw_by_canon: dict[str, str] = {}

    for header, value in raw_row.items():
        canon = canonical_header(header)
        clean = normalize_space(value)
        raw_by_canon[canon] = clean
        if canon in MASTER_COLUMNS or canon in METRIC_FIELDS:
            base[canon] = clean
        elif canon == "single AP50" or canon == "TTA AP50" or canon in {"single mean", "TTA mean"}:
            base[canon] = clean

    for key in ("config", "checkpoint", "dataset", "model", "split", "head", "status"):
        if key not in base or not normalize_space(base.get(key)):
            meta_val = metadata_value(metadata, key, key.replace("_", " "))
            if meta_val != "NA":
                base[key] = meta_val

    if "dataset" not in base or base.get("dataset") in {"", "NA"}:
        base["dataset"] = infer_dataset(file_context, metadata_value(metadata, "dataset", "data_path", "data path"))
    if "model" not in base or base.get("model") in {"", "NA"}:
        model = infer_model(file_context, base.get("checkpoint", ""), base.get("config", ""))
        if model != "NA":
            base["model"] = model
    if "angle" in base:
        base["angle"] = infer_angle(base["angle"])
    if "prompt_set" not in base or base.get("prompt_set") in {"", "NA"}:
        prompt = infer_prompt_from_heading(table.heading)
        if prompt != "NA":
            base["prompt_set"] = prompt
    if "setting" not in base or base.get("setting") in {"", "NA"}:
        setting = infer_setting_from_heading(table.heading)
        if setting != "NA":
            base["setting"] = setting
    if "head" not in base or base.get("head") in {"", "NA"}:
        if re.search(r"\balignment\b", file_context, re.I):
            base["head"] = "alignment"
        elif re.search(r"\bfusion\b", file_context, re.I):
            base["head"] = "fusion"

    log_values = []
    pred_values = []
    for header, value in raw_row.items():
        canon = canonical_header(header)
        clean = normalize_space(value)
        lower = normalize_key(header)
        if canon == "log" or lower in {"stdout", "stderr", "single log", "tta log", "eval log", "validation log"}:
            if clean:
                log_values.append(clean)
        if canon == "predictions" or "prediction" in lower:
            if clean:
                pred_values.append(clean)
    if log_values:
        base["log"] = "; ".join(dict.fromkeys(log_values))
    if pred_values:
        base["predictions"] = "; ".join(dict.fromkeys(pred_values))

    expanded: list[dict[str, Any]] = []
    specials: list[tuple[str, str, str]] = []
    if base.get("single AP50"):
        specials.append(("single_view", "AP50", base["single AP50"]))
    if base.get("TTA AP50"):
        specials.append(("rotation_tta", "AP50", base["TTA AP50"]))
    if base.get("single mean"):
        specials.append(("single_view", "mAP", base["single mean"]))
    if base.get("TTA mean"):
        specials.append(("rotation_tta", "mAP", base["TTA mean"]))

    if specials and "AP50" not in base and "mAP" not in base:
        for suffix, metric, value in specials:
            row = dict(base)
            existing = normalize_space(row.get("setting", ""))
            row["setting"] = suffix if not existing or existing == "NA" else f"{existing};{suffix}"
            row[metric] = value
            expanded.append(row)
    else:
        expanded.append(base)

    for row in expanded:
        for col in MASTER_COLUMNS:
            row.setdefault(col, "NA")
        row["status"] = clean_status(row.get("status", "NA"))
        for metric in METRIC_FIELDS:
            if metric in row and normalize_space(row[metric]) not in {"", "NA"}:
                val = parse_float(row[metric])
                if val is not None:
                    row[metric] = f"{val:.4f}"
        if row.get("angle") not in {"", "NA"}:
            row["angle"] = infer_angle(row["angle"])
    return expanded


def is_summary_file(path: Path, first_heading: str) -> bool:
    name = path.name.lower()
    heading = first_heading.lower()
    return "summary" in name or "summary" in heading or "master" in heading


def parse_source_file(path: Path, work_dir: Path) -> SourceFile:
    text, encoding = read_text_with_fallback(path)
    stat = path.stat()
    metadata = parse_bullet_metadata(text)
    tables, table_warnings = parse_tables(text)
    first_heading = extract_first_heading(text)
    important_paths = extract_paths(text)
    rel = str(path.relative_to(work_dir))
    dummy = SourceFile(
        path=path,
        relative_path=rel,
        filename=path.name,
        mtime=dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        mtime_ts=stat.st_mtime,
        size=stat.st_size,
        sha256=sha256_file(path),
        first_heading=first_heading,
        text=text,
        metadata=metadata,
        generated_at=parse_generated_at(text, metadata),
        out_root=metadata_value(metadata, "out_root", "work_dir", "out dir", "output dir"),
        result_csv=metadata_value(metadata, "result_csv", "csv", "results csv"),
        tables=tables,
        important_paths=important_paths,
        rows=[],
        warnings=[f"{rel}: decoded with fallback {encoding}"] if encoding == "utf-8-replace" else [],
        is_summary=False,
    )
    dummy.is_summary = is_summary_file(path, first_heading)
    dummy.warnings.extend(f"{rel}: {w}" for w in table_warnings)
    rows: list[dict[str, Any]] = []
    if not dummy.is_summary:
        for table in tables:
            for raw_row in table.rows:
                rows.extend(normalize_row_from_table(dummy, table, raw_row, rel, path.name))
    else:
        dummy.warnings.append(f"{rel}: summary-like file kept as source reference; tables not added to master rows")
    dummy.rows = rows
    return dummy


def should_skip(path: Path, out_md: Path, exclude_readme: bool) -> str | None:
    name = path.name
    lower = name.lower()
    try:
        if path.resolve() == out_md.resolve():
            return "out_md"
    except FileNotFoundError:
        if path.absolute() == out_md.absolute():
            return "out_md"
    if exclude_readme and "readme" in lower:
        return "README"
    if name.startswith(".") or name.endswith("~") or lower.endswith(".bak.md") or lower.endswith(".tmp.md"):
        return "temporary"
    try:
        if path.stat().st_size == 0:
            return "empty"
    except OSError:
        return "unreadable"
    return None


def scan_files(work_dir: Path, out_md: Path, recursive: bool, exclude_readme: bool) -> tuple[list[Path], list[dict[str, str]]]:
    iterator = work_dir.rglob("*.md") if recursive else work_dir.glob("*.md")
    sources: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        reason = should_skip(path, out_md, exclude_readme)
        if reason:
            skipped.append({"reason": reason, "file": str(path)})
        else:
            sources.append(path)
    return sources, skipped


def dedup_files(files: list[SourceFile]) -> tuple[list[SourceFile], list[dict[str, Any]], int]:
    by_hash: dict[str, list[SourceFile]] = defaultdict(list)
    for sf in files:
        by_hash[sf.sha256].append(sf)
    kept: list[SourceFile] = []
    duplicate_groups: list[dict[str, Any]] = []
    skipped_count = 0
    for sha, group in by_hash.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(group, key=lambda x: (x.mtime_ts, x.relative_path), reverse=True)
        primary = group_sorted[0]
        skipped = group_sorted[1:]
        kept.append(primary)
        skipped_count += len(skipped)
        duplicate_groups.append(
            {
                "sha256": sha,
                "kept_file": primary.relative_path,
                "skipped_files": [s.relative_path for s in skipped],
                "reason": "identical sha256; kept newest mtime",
            }
        )
    return sorted(kept, key=lambda x: x.relative_path), duplicate_groups, skipped_count


def file_level_metadata(sf: SourceFile, rows: list[dict[str, Any]]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in EXPERIMENT_KEY_FIELDS:
        val = metadata_value(sf.metadata, key, key.replace("_", " "))
        if val != "NA":
            meta[key] = val
    for field_name in ("dataset", "model", "config", "checkpoint", "split", "angle", "setting", "prompt_set", "head"):
        if field_name in meta:
            continue
        vals = [normalize_space(r.get(field_name, "")) for r in rows if normalize_space(r.get(field_name, "")) not in {"", "NA"}]
        if vals:
            meta[field_name] = Counter(vals).most_common(1)[0][0]
    if "dataset" not in meta:
        ds = infer_dataset(sf.first_heading, sf.filename, sf.text[:1000])
        if ds != "NA":
            meta["dataset"] = ds
    if "model" not in meta:
        model = infer_model(sf.first_heading, sf.filename, sf.text[:1000])
        if model != "NA":
            meta["model"] = model
    if sf.out_root != "NA":
        meta.setdefault("out_root", sf.out_root)
    if sf.result_csv != "NA":
        meta.setdefault("result_csv", sf.result_csv)
    return meta


def experiment_key_for(sf: SourceFile) -> str:
    meta = file_level_metadata(sf, sf.rows)
    parts = [f"{k}={normalize_space(meta[k])}" for k in EXPERIMENT_KEY_FIELDS if normalize_space(meta.get(k, ""))]
    if parts:
        return " | ".join(parts)
    fallback = [sf.first_heading, re.sub(r"\d{8}_\d{6}", "", sf.filename), sf.result_csv]
    preds = sf.important_paths.get("prediction_files", [])
    if preds:
        fallback.append(str(Path(preds[0]).parent))
    return "fallback | " + " | ".join(normalize_space(x) for x in fallback if x and x != "NA")


def dedup_experiments(files: list[SourceFile]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[SourceFile]] = defaultdict(list)
    for sf in files:
        groups[experiment_key_for(sf)].append(sf)
    report: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda x: (x.is_summary, -x.mtime_ts, x.relative_path))
        non_summary = [g for g in group if not g.is_summary]
        primary_pool = non_summary if non_summary else group
        primary = sorted(primary_pool, key=lambda x: (x.mtime_ts, x.relative_path), reverse=True)[0]
        row_sigs = {tuple(metric_signature(r) for r in sf.rows) for sf in group if sf.rows}
        status = "UNIQUE" if len(group) == 1 else "MERGED"
        if len(row_sigs) > 1:
            status = "CONFLICT"
            conflicts.append(
                {
                    "conflict_key": key,
                    "source_file": "; ".join(sf.relative_path for sf in group),
                    "metric_fields": "file-level row metric signatures",
                    "values": f"{len(row_sigs)} distinct metric signatures",
                }
            )
        report.append(
            {
                "experiment_key": key,
                "kept_primary": primary.relative_path,
                "merged_sources": [sf.relative_path for sf in group],
                "status": status,
            }
        )
    return report, conflicts


def row_key_for(row: dict[str, Any]) -> str:
    parts = []
    missing = 0
    for field_name in ROW_KEY_FIELDS:
        value = normalize_space(row.get(field_name, ""))
        if not value or value == "NA":
            missing += 1
            continue
        parts.append(f"{field_name}={value.lower()}")
    if len(parts) >= 3:
        return " | ".join(parts)
    norm_row = "|".join(f"{k}={normalize_space(v).lower()}" for k, v in sorted(row.items()) if k not in {"source_file", "source_table"})
    return "row_hash=" + stable_hash(norm_row)


def dedup_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    row_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_groups[row_key_for(row)].append(row)

    duplicate_report: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, group in sorted(row_groups.items()):
        sig_groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            sig_groups[metric_signature(row)].append(row)
        if len(group) == 1:
            row = dict(group[0])
            row["duplicate_status"] = "UNIQUE"
            kept.append(row)
            continue
        if len(sig_groups) == 1:
            primary = sorted(group, key=lambda r: normalize_space(r.get("source_file", "")))[0]
            row = dict(primary)
            row["duplicate_status"] = "DEDUPED"
            kept.append(row)
            duplicate_report.append(
                {
                    "row_key": key,
                    "kept_source": primary.get("source_file", "NA"),
                    "duplicate_sources": sorted({r.get("source_file", "NA") for r in group if r is not primary}),
                    "status": "DEDUPED_IDENTICAL_METRICS",
                }
            )
        else:
            for sig, rows_for_sig in sig_groups.items():
                primary = sorted(rows_for_sig, key=lambda r: normalize_space(r.get("source_file", "")))[0]
                row = dict(primary)
                row["duplicate_status"] = "CONFLICT"
                row["note"] = "same key but different metrics"
                kept.append(row)
            fields = sorted({field_name for sig in sig_groups for field_name, _ in sig})
            values = {
                str(sig): sorted({r.get("source_file", "NA") for r in rows_for_sig})
                for sig, rows_for_sig in sig_groups.items()
            }
            conflicts.append(
                {
                    "conflict_key": key,
                    "source_file": "; ".join(sorted({r.get("source_file", "NA") for r in group})),
                    "metric_fields": ", ".join(fields) if fields else "no metric fields",
                    "values": values,
                }
            )
            duplicate_report.append(
                {
                    "row_key": key,
                    "kept_source": "multiple",
                    "duplicate_sources": sorted({r.get("source_file", "NA") for r in group}),
                    "status": "CONFLICT",
                }
            )
    return kept, duplicate_report, conflicts


def aggregate_important_paths(files: list[SourceFile]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for sf in files:
        for cat, paths in sf.important_paths.items():
            for path in paths:
                key = (cat, path, sf.relative_path)
                if key in seen:
                    continue
                seen.add(key)
                out[cat].append({"path": path, "source_file": sf.relative_path})
    return {k: sorted(v, key=lambda x: (x["path"], x["source_file"])) for k, v in out.items()}


def most_common_values(rows: list[dict[str, Any]], field_name: str, n: int = 10) -> list[tuple[str, int]]:
    vals = [normalize_space(r.get(field_name, "")) for r in rows if normalize_space(r.get(field_name, "")) not in {"", "NA"}]
    return Counter(vals).most_common(n)


def angle_sweep_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        angle = normalize_space(row.get("angle", ""))
        if not angle or angle == "NA":
            continue
        metric = "mAP" if parse_float(row.get("mAP")) is not None else "AP50" if parse_float(row.get("AP50")) is not None else None
        if metric is None:
            continue
        key = (
            normalize_space(row.get("dataset", "NA")),
            normalize_space(row.get("model", "NA")),
            normalize_space(row.get("setting", "NA")),
            normalize_space(row.get("prompt_set", "NA")),
            normalize_space(row.get("head", "NA")),
        )
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        metric = "mAP" if any(parse_float(r.get("mAP")) is not None for r in group) else "AP50"
        by_angle: dict[str, float] = {}
        for row in group:
            val = parse_float(row.get(metric))
            if val is None:
                continue
            by_angle[normalize_space(row.get("angle", "NA"))] = val
        if len(by_angle) < 2:
            continue
        values = list(by_angle.values())
        mean = sum(values) / len(values)
        min_angle, min_val = min(by_angle.items(), key=lambda item: item[1])
        max_angle, max_val = max(by_angle.items(), key=lambda item: item[1])
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        summaries.append(
            {
                "group": " | ".join(f for f in key if f and f != "NA") or "NA",
                "n_angles": len(by_angle),
                "metric_used": metric,
                "mean": f"{mean:.4f}",
                "min": f"{min_val:.4f}",
                "worst_angle": min_angle,
                "max": f"{max_val:.4f}",
                "best_angle": max_angle,
                "range": f"{(max_val - min_val):.4f}",
                "std": f"{std:.4f}",
            }
        )
    return summaries


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        status = normalize_space(row.get("status", "NA"))
        if status == "NA" or status.lower() in STATUS_OK:
            continue
        out.append(
            {
                "status": status,
                "dataset": normalize_space(row.get("dataset", "NA")),
                "model": normalize_space(row.get("model", "NA")),
                "split": normalize_space(row.get("split", "NA")),
                "angle": normalize_space(row.get("angle", "NA")),
                "source_file": normalize_space(row.get("source_file", "NA")),
                "note": normalize_space(row.get("note", "")) or "NA",
            }
        )
    return out


def table(headers: list[str], rows: list[dict[str, Any] | list[Any]]) -> str:
    out = ["| " + " | ".join(md_escape(h) for h in headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        if isinstance(row, dict):
            vals = [row.get(h, "NA") for h in headers]
        else:
            vals = list(row)
        out.append("| " + " | ".join(md_escape(v) for v in vals) + " |")
    return "\n".join(out)


def build_high_level_findings(rows: list[dict[str, Any]], sweep: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    models = most_common_values(rows, "model", 5)
    datasets = most_common_values(rows, "dataset", 5)
    angles = most_common_values(rows, "angle", 8)
    statuses = most_common_values(rows, "status", 10)
    if models:
        findings.append("Most frequent models: " + ", ".join(f"{m} ({c})" for m, c in models) + ".")
    if datasets:
        findings.append("Most frequent datasets: " + ", ".join(f"{d} ({c})" for d, c in datasets) + ".")
    if angles:
        findings.append("Most frequent angles: " + ", ".join(f"{a} ({c})" for a, c in angles) + ".")
    bad_statuses = [(s, c) for s, c in statuses if s.lower() not in STATUS_OK and s != "NA"]
    if bad_statuses:
        findings.append("Non-DONE/OK statuses detected: " + ", ".join(f"{s} ({c})" for s, c in bad_statuses) + ".")
    if sweep:
        best = sorted(sweep, key=lambda x: parse_float(x.get("range")) or 0.0, reverse=True)[:5]
        findings.append("Angle sweep groups parsed: " + str(len(sweep)) + ". Largest ranges: " + "; ".join(f"{b['group']} range={b['range']}" for b in best) + ".")
    if not findings:
        findings.append("Insufficient structured metrics for automatic conclusion.")
    return findings


def condensed_summary(sf: SourceFile) -> list[str]:
    rows = sf.rows
    models = ", ".join(m for m, _ in most_common_values(rows, "model", 5)) or infer_model(sf.first_heading, sf.filename)
    datasets = ", ".join(d for d, _ in most_common_values(rows, "dataset", 5)) or infer_dataset(sf.first_heading, sf.filename)
    metric_fields = sorted({m for r in rows for m in METRIC_FIELDS if normalize_space(r.get(m, "")) not in {"", "NA"}})
    failures = [r for r in rows if normalize_space(r.get("status", "NA")).lower() not in STATUS_OK and normalize_space(r.get("status", "NA")) != "NA"]
    imp = []
    for cat, paths in sf.important_paths.items():
        if paths:
            imp.append(f"{cat}: {len(paths)}")
    return [
        f"* title: {md_escape(sf.first_heading)}",
        f"* generated_at: {md_escape(sf.generated_at)}",
        f"* out_root: {md_escape(sf.out_root)}",
        f"* result_csv: {md_escape(sf.result_csv)}",
        f"* key models: {md_escape(models or 'NA')}",
        f"* key datasets: {md_escape(datasets or 'NA')}",
        f"* key metrics: {md_escape(', '.join(metric_fields) or 'NA')}",
        f"* failures: {len(failures)}",
        f"* source tables parsed: {len(sf.tables)}",
        f"* important paths: {md_escape(', '.join(imp) or 'NA')}",
    ]


def build_markdown(
    *,
    work_dir: Path,
    out_md: Path,
    recursive: bool,
    source_count: int,
    skipped_files: list[dict[str, str]],
    kept_files: list[SourceFile],
    duplicate_file_groups: list[dict[str, Any]],
    duplicate_files_skipped: int,
    experiment_groups: list[dict[str, Any]],
    row_duplicates: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    warnings: list[str],
    important_paths: dict[str, list[dict[str, str]]],
    sweep_groups: list[dict[str, Any]],
) -> str:
    git_hash = get_git_hash(Path.cwd())
    skipped_counter = Counter(s["reason"] for s in skipped_files)
    findings = build_high_level_findings(rows, sweep_groups)
    failure = failure_rows(rows)

    lines: list[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## 1. Metadata")
    lines.append("")
    meta_rows = [
        ["generated_at", now_text()],
        ["work_dir", str(work_dir)],
        ["out_md", str(out_md)],
        ["recursive", str(recursive)],
        ["total md files found", source_count],
        ["skipped README files", skipped_counter.get("README", 0)],
        ["skipped temporary files", skipped_counter.get("temporary", 0)],
        ["duplicate files skipped", duplicate_files_skipped],
        ["unique experiment records", len(experiment_groups)],
        ["unique result rows", len(rows)],
        ["conflict count", len(conflicts)],
        ["warning count", len(warnings)],
        ["script path", str(SCRIPT_PATH)],
        ["git commit hash", git_hash],
    ]
    lines.append(table(["key", "value"], meta_rows))
    lines.append("")

    lines.append("## 2. Source Files")
    lines.append("")
    source_rows = [
        {
            "kept": "yes",
            "file": sf.filename,
            "relative_path": sf.relative_path,
            "mtime": sf.mtime,
            "size": sf.size,
            "sha256_short": sf.sha256[:12],
            "first_heading": sf.first_heading,
        }
        for sf in kept_files
    ]
    lines.append(table(["kept", "file", "relative_path", "mtime", "size", "sha256_short", "first_heading"], source_rows))
    lines.append("")
    lines.append("Skipped files:")
    lines.append("")
    skipped_rows = []
    for item in skipped_files:
        if item.get("reason") == "out_md":
            skipped_rows.append({"reason": "out_md", "file": "<output md self-excluded>"})
        else:
            skipped_rows.append(item)
    lines.append(table(["reason", "file"], skipped_rows or [{"reason": "NA", "file": "NA"}]))
    lines.append("")

    lines.append("## 3. Deduplication Report")
    lines.append("")
    lines.append("### 3.1 File-level duplicates")
    lines.append("")
    dup_file_rows = []
    for group in duplicate_file_groups:
        for skipped in group["skipped_files"]:
            dup_file_rows.append({"kept_file": group["kept_file"], "skipped_file": skipped, "reason": group["reason"]})
    lines.append(table(["kept_file", "skipped_file", "reason"], dup_file_rows or [{"kept_file": "NA", "skipped_file": "NA", "reason": "No duplicate files found."}]))
    lines.append("")
    lines.append("### 3.2 Experiment-level duplicates")
    lines.append("")
    exp_rows = [
        {
            "experiment_key": g["experiment_key"],
            "kept_primary": g["kept_primary"],
            "merged_sources": "; ".join(g["merged_sources"]),
            "status": g["status"],
        }
        for g in experiment_groups
    ]
    lines.append(table(["experiment_key", "kept_primary", "merged_sources", "status"], exp_rows))
    lines.append("")
    lines.append("### 3.3 Row-level duplicates")
    lines.append("")
    lines.append(table(["row_key", "kept_source", "duplicate_sources", "status"], row_duplicates or [{"row_key": "NA", "kept_source": "NA", "duplicate_sources": "NA", "status": "No row duplicates found."}]))
    lines.append("")
    lines.append("### 3.4 Conflicts")
    lines.append("")
    if conflicts:
        conflict_rows = [
            {
                "conflict_key": c["conflict_key"],
                "source_file": c["source_file"],
                "metric_fields": c["metric_fields"],
                "values": json.dumps(c["values"], ensure_ascii=False) if isinstance(c.get("values"), (dict, list)) else c.get("values", "NA"),
            }
            for c in conflicts
        ]
        lines.append(table(["conflict_key", "source_file", "metric_fields", "values"], conflict_rows))
    else:
        lines.append("No conflicts found.")
    lines.append("")

    lines.append("## 4. High-level Findings")
    lines.append("")
    for idx, finding in enumerate(findings, 1):
        lines.append(f"{idx}. {finding}")
    lines.append("")

    lines.append("## 5. Master Result Table")
    lines.append("")
    master_rows = []
    for row in rows:
        out_row = {}
        for col in MASTER_COLUMNS:
            value = row.get(col, "NA")
            if col in METRIC_FIELDS:
                value = fmt_num(value)
            out_row[col] = value
        master_rows.append(out_row)
    lines.append(table(MASTER_COLUMNS, master_rows or [{c: "NA" for c in MASTER_COLUMNS}]))
    lines.append("")

    lines.append("## 6. Angle Sweep Summaries")
    lines.append("")
    lines.append(table(["group", "n_angles", "metric_used", "mean", "min", "worst_angle", "max", "best_angle", "range", "std"], sweep_groups or [{"group": "NA", "n_angles": 0, "metric_used": "NA", "mean": "NA", "min": "NA", "worst_angle": "NA", "max": "NA", "best_angle": "NA", "range": "NA", "std": "NA"}]))
    lines.append("")

    lines.append("## 7. Failure / Partial Records")
    lines.append("")
    lines.append(table(["status", "dataset", "model", "split", "angle", "source_file", "note"], failure or [{"status": "NA", "dataset": "NA", "model": "NA", "split": "NA", "angle": "NA", "source_file": "NA", "note": "No failure or partial records parsed from structured rows."}]))
    lines.append("")

    lines.append("## 8. Important Paths")
    lines.append("")
    path_section_map = [
        ("8.1 prediction files", "prediction_files"),
        ("8.2 log files", "log_files"),
        ("8.3 configs", "configs"),
        ("8.4 checkpoints", "checkpoints"),
        ("8.5 csv / tsv result files", "csv_tsv_result_files"),
    ]
    for title, cat in path_section_map:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(table(["path", "source_file"], important_paths.get(cat, []) or [{"path": "NA", "source_file": "NA"}]))
        lines.append("")

    lines.append("## 9. Per-file Condensed Summaries")
    lines.append("")
    for sf in kept_files:
        lines.append(f"### {sf.relative_path}")
        lines.append("")
        lines.extend(condensed_summary(sf))
        lines.append("")

    lines.append("## 10. Appendix: Unparsed Content Notices")
    lines.append("")
    notice_rows = []
    for sf in kept_files:
        if not sf.tables:
            notice_rows.append({"file": sf.relative_path, "reason": "No Markdown tables parsed."})
        if not sf.rows and not sf.is_summary:
            notice_rows.append({"file": sf.relative_path, "reason": "No structured result rows parsed."})
        if sf.is_summary:
            notice_rows.append({"file": sf.relative_path, "reason": "Summary-like file kept as reference; table rows not added to Master Result Table."})
    for warning in warnings:
        notice_rows.append({"file": "warning", "reason": warning})
    lines.append(table(["file", "reason"], notice_rows or [{"file": "NA", "reason": "All kept files had parseable structured content."}]))
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    source_files: list[SourceFile],
    skipped_files: list[dict[str, str]],
    duplicate_file_groups: list[dict[str, Any]],
    experiment_groups: list[dict[str, Any]],
    row_duplicates: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    warnings: list[str],
    important_paths: dict[str, list[dict[str, str]]],
    angle_sweeps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_files": [
            {
                "file_path": str(sf.path),
                "relative_path": sf.relative_path,
                "filename": sf.filename,
                "mtime": sf.mtime,
                "size": sf.size,
                "sha256": sf.sha256,
                "first_heading": sf.first_heading,
                "generated_at": sf.generated_at,
                "out_root": sf.out_root,
                "result_csv": sf.result_csv,
                "is_summary": sf.is_summary,
            }
            for sf in source_files
        ],
        "skipped_files": skipped_files,
        "duplicate_file_groups": duplicate_file_groups,
        "experiment_groups": experiment_groups,
        "row_duplicates": row_duplicates,
        "conflicts": conflicts,
        "warnings": warnings,
        "important_paths": important_paths,
        "angle_sweep_groups": angle_sweeps,
    }


def report_path_for(out_md: Path) -> Path:
    return out_md.with_name(out_md.stem + "_dedup_report.json")


def run(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).resolve()
    out_md = Path(args.out_md).resolve()
    recursive = parse_bool(args.recursive)
    exclude_readme = parse_bool(args.exclude_readme)
    if not work_dir.exists() or not work_dir.is_dir():
        raise SystemExit(f"work-dir does not exist or is not a directory: {work_dir}")

    source_paths, skipped_files = scan_files(work_dir, out_md, recursive, exclude_readme)
    parsed_files: list[SourceFile] = []
    parse_warnings: list[str] = []
    for path in source_paths:
        try:
            sf = parse_source_file(path, work_dir)
            parsed_files.append(sf)
            parse_warnings.extend(sf.warnings)
        except Exception as exc:
            skipped_files.append({"reason": f"parse_error: {exc}", "file": str(path)})
            parse_warnings.append(f"{path}: parse error: {exc}")

    kept_files, duplicate_file_groups, duplicate_files_skipped = dedup_files(parsed_files)
    experiment_groups, experiment_conflicts = dedup_experiments(kept_files)
    all_rows = [row for sf in kept_files for row in sf.rows]
    deduped_rows, row_duplicates, row_conflicts = dedup_rows(all_rows)
    conflicts = experiment_conflicts + row_conflicts
    important_paths = aggregate_important_paths(kept_files)
    sweep_groups = angle_sweep_groups(deduped_rows)

    readme_skipped = sum(1 for s in skipped_files if s["reason"] == "README")
    temp_skipped = sum(1 for s in skipped_files if s["reason"] == "temporary")

    if args.verbose or args.dry_run:
        print(f"source_md_files_scanned: {len(source_paths)}")
        print(f"readme_files_skipped: {readme_skipped}")
        print(f"temporary_files_skipped: {temp_skipped}")
        print(f"out_md_excluded: {sum(1 for s in skipped_files if s['reason'] == 'out_md')}")
        print(f"file_duplicate_candidates: {len(duplicate_file_groups)}")
        print(f"experiment_duplicate_candidates: {sum(1 for g in experiment_groups if len(g['merged_sources']) > 1)}")
        print(f"row_duplicate_candidates: {len(row_duplicates)}")
        print(f"conflict_candidates: {len(conflicts)}")
    if args.dry_run:
        print("DRY_RUN_SOURCE_FILES")
        for sf in kept_files:
            print(sf.relative_path)
        print("DRY_RUN_SKIPPED_FILES")
        for item in skipped_files:
            print(f"{item['reason']}: {item['file']}")
        print("DRY_RUN_DONE_NO_FILES_WRITTEN")
        return 0

    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_md.exists():
        backup = out_md.with_name(out_md.name + f".bak.{ts_for_backup()}")
        shutil.copy2(out_md, backup)
        if args.verbose:
            print(f"backup_created: {backup}")

    markdown = build_markdown(
        work_dir=work_dir,
        out_md=out_md,
        recursive=recursive,
        source_count=len(source_paths) + len(skipped_files),
        skipped_files=skipped_files,
        kept_files=kept_files,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_files_skipped=duplicate_files_skipped,
        experiment_groups=experiment_groups,
        row_duplicates=row_duplicates,
        conflicts=conflicts,
        rows=deduped_rows,
        warnings=parse_warnings,
        important_paths=important_paths,
        sweep_groups=sweep_groups,
    )
    out_md.write_text(markdown, encoding="utf-8")

    dedup_report_path = report_path_for(out_md)
    if args.write_dedup_report:
        report = build_report(
            source_files=kept_files,
            skipped_files=skipped_files,
            duplicate_file_groups=duplicate_file_groups,
            experiment_groups=experiment_groups,
            row_duplicates=row_duplicates,
            conflicts=conflicts,
            warnings=parse_warnings,
            important_paths=important_paths,
            angle_sweeps=sweep_groups,
        )
        dedup_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("MERGE_EXPERIMENT_MD_DONE")
    print(f"source_md_files_scanned: {len(source_paths)}")
    print(f"readme_files_skipped: {readme_skipped}")
    print(f"temporary_files_skipped: {temp_skipped}")
    print(f"duplicate_files_skipped: {duplicate_files_skipped}")
    print(f"unique_experiment_groups: {len(experiment_groups)}")
    print(f"unique_result_rows: {len(deduped_rows)}")
    print(f"conflicts: {len(conflicts)}")
    print(f"warnings: {len(parse_warnings)}")
    print("")
    print(f"output_md: {out_md}")
    print(f"dedup_report: {dedup_report_path if args.write_dedup_report else 'NA'}")
    print(f"script: {SCRIPT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/data1/zcy/OpenRSD/resultmd")
    parser.add_argument("--out-md", default="/data1/zcy/OpenRSD/resultmd/summary_all_experiments_dedup.md")
    parser.add_argument("--recursive", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--exclude-readme", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep-conflicts", action="store_true", default=True)
    parser.add_argument("--write-dedup-report", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

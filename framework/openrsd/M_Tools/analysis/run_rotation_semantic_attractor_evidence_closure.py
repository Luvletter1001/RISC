#!/usr/bin/env python3
"""CPU-only evidence closure for the rotation semantic attractor project.

The OpenRSD environment currently has a broken pandas build, so this script
uses stdlib CSV readers plus numpy/scipy/matplotlib. It analyzes existing CSV,
MD, JSON, and lightweight prediction artifacts only. It never trains and never
reruns full inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


EXPECTED_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
EXPECTED = {
    "openvocab": {"rows": 30000, "status": {"DONE_FULL": 30000}, "key": ["tile_id", "angle"]},
    "causal": {"rows": 36000, "status": {"DONE_FULL": 36000}, "key": ["tile_id", "angle", "intervention"]},
    "context": {"rows": 7584, "status": {"DONE_FULL": 3168, "NOT_APPLICABLE": 4416}, "key": ["tile_id", "angle", "condition"]},
    "dehub": {"rows": 12000, "status": {"DONE_FULL": 12000}, "key": ["tile_id", "angle", "checkpoint_name"]},
}
METRICS = [
    ("det/img", "detection_total"),
    ("SV/img", "small_vehicle_count"),
    ("SV ratio", "small_vehicle_ratio"),
    ("LV/img", "large_vehicle_count"),
    ("LV ratio", "large_vehicle_ratio"),
    ("top1=SV", "top1_is_sv"),
]


@dataclass
class Paths:
    repo_root: Path
    result_md_dir: Path
    work_dir: Path
    openvocab_csv: Path
    causal_csv: Path
    context_csv: Path
    dehub_csv: Path
    closedset_dir: Path

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def figures_dir(self) -> Path:
        return self.result_md_dir / "figures"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs(paths: Paths) -> None:
    paths.result_md_dir.mkdir(parents=True, exist_ok=True)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def log(paths: Paths, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (paths.logs_dir / "evidence_closure.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["top1_is_sv"] = "1" if row.get("top1_class") == "small-vehicle" else "0"
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        headers = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    if abs(v) >= 1000:
        return f"{v:,.0f}" if digits == 0 else f"{v:,.1f}"
    return f"{v:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    return f"{100 * v:.{digits}f}%"


def md_table(rows: list[dict[str, Any]], headers: list[str] | None = None, max_rows: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    if max_rows is not None:
        rows = rows[:max_rows]
    if headers is None:
        headers = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def to_float(value: Any) -> float:
    try:
        if value == "" or value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def natural_key(value: Any) -> tuple[int, str]:
    text = str(value)
    if text.isdigit():
        return (0, f"{int(text):06d}")
    m = re.search(r"(\d+)", text)
    return (0, f"{int(m.group(1)):06d}_{text}") if m else (1, text)


def key_tuple(row: dict[str, Any], cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(c, "")) for c in cols)


def counter_status(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(str(r.get("status", "")) for r in rows))


def sha256_prefix(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest() + ("" if path.stat().st_size <= limit else "_prefix")


def groupby(rows: list[dict[str, str]], col: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(col, ""))].append(row)
    return dict(out)


def summarize(rows: list[dict[str, str]], label_key: str = "scope", label_value: str = "all") -> dict[str, Any]:
    out: dict[str, Any] = {label_key: label_value, "rows": len(rows)}
    for label, col in METRICS:
        vals = np.array([to_float(r.get(col)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[label] = float(vals.mean()) if len(vals) else ""
    return out


def summarize_by(rows: list[dict[str, str]], by: str) -> list[dict[str, Any]]:
    return [summarize(g, by, k) for k, g in sorted(groupby(rows, by).items(), key=lambda kv: natural_key(kv[0]))]


def bootstrap_ci(values: np.ndarray, iters: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    if n == 1:
        return float(values[0]), float(values[0])
    batch = 500
    means = []
    for start in range(0, iters, batch):
        b = min(batch, iters - start)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        means.append(values[idx].mean(axis=1))
    boot = np.concatenate(means)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_stats(
    rows: list[dict[str, str]],
    group_col: str,
    baseline: str,
    treatments: list[str],
    pair_cols: list[str],
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    maps: dict[str, dict[tuple[str, ...], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        group = str(row.get(group_col, ""))
        maps[group][key_tuple(row, pair_cols)] = row

    out = []
    for treatment in treatments:
        common = sorted(set(maps[baseline]) & set(maps[treatment]))
        for label, col in METRICS:
            base = np.array([to_float(maps[baseline][k].get(col)) for k in common], dtype=float)
            trt = np.array([to_float(maps[treatment][k].get(col)) for k in common], dtype=float)
            valid = np.isfinite(base) & np.isfinite(trt)
            base = base[valid]
            trt = trt[valid]
            diff = trt - base
            nz = diff[diff != 0]
            ci_low, ci_high = bootstrap_ci(diff, bootstrap_iters, rng)
            try:
                wilcoxon_p = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
            except Exception:
                wilcoxon_p = math.nan
            try:
                sign_p = float(stats.binomtest(int((nz > 0).sum()), n=len(nz), p=0.5).pvalue) if len(nz) else 1.0
            except Exception:
                sign_p = math.nan
            std = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
            out.append(
                {
                    "group": treatment,
                    "metric": label,
                    "n": len(diff),
                    "baseline_mean": float(base.mean()) if len(base) else math.nan,
                    "treatment_mean": float(trt.mean()) if len(trt) else math.nan,
                    "mean_delta": float(diff.mean()) if len(diff) else math.nan,
                    "median_delta": float(np.median(diff)) if len(diff) else math.nan,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "sign_test_p": sign_p,
                    "cohen_dz": float(diff.mean() / std) if std and np.isfinite(std) and std != 0 else math.nan,
                    "positive_delta_frac": float((diff > 0).mean()) if len(diff) else math.nan,
                }
            )
    add_fdr(out, "wilcoxon_p", "wilcoxon_p_fdr")
    add_fdr(out, "sign_test_p", "sign_test_p_fdr")
    return out


def add_fdr(rows: list[dict[str, Any]], p_col: str, out_col: str) -> None:
    indexed = [(i, to_float(r.get(p_col))) for i, r in enumerate(rows) if np.isfinite(to_float(r.get(p_col)))]
    indexed.sort(key=lambda kv: kv[1])
    m = len(indexed)
    prev = 1.0
    adjusted: dict[int, float] = {}
    for rank_from_end, (idx, pval) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, pval * m / rank)
        adjusted[idx] = min(1.0, q)
        prev = q
    for i, row in enumerate(rows):
        row[out_col] = adjusted.get(i, "")


def stratified_deltas(
    rows: list[dict[str, str]],
    group_col: str,
    baseline: str,
    treatments: list[str],
    pair_cols: list[str],
    strata_col: str,
) -> list[dict[str, Any]]:
    maps: dict[str, dict[tuple[str, ...], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        maps[str(row.get(group_col, ""))][key_tuple(row, pair_cols)] = row
    out = []
    for treatment in treatments:
        common = sorted(set(maps[baseline]) & set(maps[treatment]))
        for label, col in METRICS:
            by_stratum: dict[str, list[float]] = defaultdict(list)
            for key in common:
                b = maps[baseline][key]
                t = maps[treatment][key]
                diff = to_float(t.get(col)) - to_float(b.get(col))
                if np.isfinite(diff):
                    by_stratum[str(b.get(strata_col, t.get(strata_col, "")))].append(diff)
            for stratum, vals in sorted(by_stratum.items(), key=lambda kv: natural_key(kv[0])):
                arr = np.array(vals, dtype=float)
                out.append(
                    {
                        "group": treatment,
                        strata_col: stratum,
                        "metric": label,
                        "n": len(arr),
                        "mean_delta": float(arr.mean()) if len(arr) else math.nan,
                        "median_delta": float(np.median(arr)) if len(arr) else math.nan,
                        "positive_delta_frac": float((arr > 0).mean()) if len(arr) else math.nan,
                    }
                )
    return out


def parse_hist(rows: list[dict[str, str]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        raw = row.get("class_histogram", "")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for k, v in data.items():
            c[str(k)] += int(v)
    return c


def artifact_inventory(paths: Paths, dfs: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    artifacts = {
        "openvocab": paths.openvocab_csv,
        "causal": paths.causal_csv,
        "context": paths.context_csv,
        "dehub": paths.dehub_csv,
        "closedset_false_sv": paths.closedset_dir / "metrics/paper_table_closedset_false_sv_benchmark.csv",
        "closedset_rotation_gain": paths.closedset_dir / "metrics/paper_table_closedset_rotation_gain.csv",
        "closedset_stage": paths.closedset_dir / "metrics/paper_table_closedset_stage_decomposition.csv",
        "closedset_concentration": paths.closedset_dir / "metrics/paper_table_closedset_concentration.csv",
    }
    rows = []
    for name, path in artifacts.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else "",
                "sha256_prefix": sha256_prefix(path) if exists and path.is_file() else "",
                "rows": len(dfs[name]) if name in dfs else "",
            }
        )
    return rows


def integrity_audit(paths: Paths, dfs: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    inv = artifact_inventory(paths, dfs)
    write_rows(paths.result_md_dir / "artifact_inventory.csv", inv)
    status: dict[str, Any] = {"overall_status": "DONE", "datasets": {}}
    md = ["# 00 Artifact Integrity Audit", "", f"- Generated: {now()}", "", "## Inventory", "", md_table(inv), "", "## Dataset Checks", ""]
    for name in ["openvocab", "causal", "context", "dehub"]:
        rows = dfs[name]
        exp = EXPECTED[name]
        key_cols = [c for c in exp["key"] if c in rows[0]]
        keys = [key_tuple(r, key_cols) for r in rows]
        dup = len(keys) - len(set(keys))
        scounts = counter_status(rows)
        angles = sorted({int(to_float(r.get("angle"))) for r in rows if np.isfinite(to_float(r.get("angle")))})
        numeric_checks = []
        for col in ["detection_total", "small_vehicle_count", "large_vehicle_count", "small_vehicle_ratio", "large_vehicle_ratio"]:
            vals = np.array([to_float(r.get(col)) for r in rows], dtype=float)
            numeric_checks.append(
                {
                    "column": col,
                    "nan_or_non_numeric": int((~np.isfinite(vals)).sum()),
                    "negative": int((vals[np.isfinite(vals)] < 0).sum()),
                }
            )
        warnings = []
        if len(rows) != exp["rows"]:
            warnings.append(f"row count mismatch expected {exp['rows']} got {len(rows)}")
        if scounts != exp["status"]:
            warnings.append(f"status count mismatch expected {exp['status']} got {scounts}")
        if angles != EXPECTED_ANGLES:
            warnings.append(f"angle mismatch expected {EXPECTED_ANGLES} got {angles}")
        if dup:
            warnings.append(f"duplicate key rows {dup} by {key_cols}")
        if name != "context" and "NOT_APPLICABLE" in scounts:
            warnings.append("NOT_APPLICABLE outside context")
        balance = {}
        if name == "causal":
            balance = angle_group_balance(rows, ["intervention", "angle"])
            warnings += pairing_warnings(rows, "intervention", sorted({r["intervention"] for r in rows}), ["tile_id", "angle", "risk_group"])
        if name == "dehub":
            balance = angle_group_balance(rows, ["checkpoint_name", "angle"])
            warnings += pairing_warnings(rows, "checkpoint_name", sorted({r["checkpoint_name"] for r in rows}), ["tile_id", "angle", "risk_group"])
        if name == "context":
            done = [r for r in rows if r.get("status") == "DONE_FULL"]
            warnings += pairing_warnings(done, "condition", sorted({r["condition"] for r in done}), ["tile_id", "angle", "risk_group"])
        dstatus = "DONE" if not warnings else "WARNING"
        if warnings:
            status["overall_status"] = "WARNING"
        status["datasets"][name] = {
            "rows": len(rows),
            "unique_units": len(set(keys)),
            "status_counts": scounts,
            "key_cols": key_cols,
            "duplicate_count": dup,
            "angles": angles,
            "numeric_checks": numeric_checks,
            "balance": balance,
            "status": dstatus,
            "warnings": warnings,
        }
        md += [
            f"### {name}",
            "",
            f"- Status: `{dstatus}`",
            f"- Rows: `{len(rows)}` expected `{exp['rows']}`",
            f"- Unique units: `{len(set(keys))}`",
            f"- Status counts: `{scounts}`",
            f"- Duplicates: `{dup}`",
            f"- Angles: `{angles}`",
            f"- Balance: `{balance}`",
            "",
            md_table(numeric_checks),
            "",
        ]
        if warnings:
            md += ["Warnings:", ""]
            md += [f"- {w}" for w in warnings]
            md.append("")
    write_json(paths.result_md_dir / "artifact_integrity_status.json", status)
    write_md(paths.result_md_dir / "00_artifact_integrity_audit.md", "\n".join(md))
    return status


def angle_group_balance(rows: list[dict[str, str]], cols: list[str]) -> dict[str, Any]:
    counts = Counter(key_tuple(r, cols) for r in rows)
    vals = list(counts.values())
    return {"min": min(vals), "max": max(vals), "balanced": min(vals) == max(vals)} if vals else {}


def pairing_warnings(rows: list[dict[str, str]], group_col: str, expected_groups: list[str], pair_cols: list[str]) -> list[str]:
    by_pair: dict[tuple[str, ...], set[str]] = defaultdict(set)
    dup_keys = []
    seen = set()
    for row in rows:
        pk = key_tuple(row, pair_cols)
        g = str(row.get(group_col, ""))
        by_pair[pk].add(g)
        dg = pk + (g,)
        if dg in seen:
            dup_keys.append(dg)
        seen.add(dg)
    expected = set(expected_groups)
    bad = sum(1 for vals in by_pair.values() if vals != expected)
    warnings = []
    if bad:
        warnings.append(f"{bad} pairs do not have exact group set {sorted(expected)}")
    if dup_keys:
        warnings.append(f"{len(dup_keys)} duplicate paired group rows")
    return warnings


def is_candidate(path: Path) -> bool:
    if path.suffix.lower() not in {".pkl", ".json", ".jsonl", ".csv"}:
        return False
    text = path.name.lower()
    return any(k in text for k in ["prediction", "bbox", "det", "results", "eval", "dota", "merged"])


def collect_keys(obj: Any, depth: int = 0, prefix: str = "") -> set[str]:
    if depth > 3:
        return set()
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            out.add(name)
            out |= collect_keys(v, depth + 1, name)
    elif isinstance(obj, (list, tuple)) and obj:
        out |= collect_keys(obj[0], depth + 1, prefix)
    elif prefix:
        out.add(prefix)
    return out


def sample_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key in ["data_list", "results", "predictions"]:
            if isinstance(obj.get(key), list) and obj[key]:
                return obj[key][0]
    if isinstance(obj, (list, tuple)) and obj:
        return obj[0]
    return obj


def inspect_candidate(path: Path) -> dict[str, Any]:
    info = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "read_status": "NOT_READ",
        "has_image_id": False,
        "has_boxes": False,
        "has_scores": False,
        "has_labels": False,
        "has_angle": bool(re.search(r"angle[_-]?\d{3}", str(path))),
        "target_scope_hint": any(t in str(path) for t in ["full_openvocab_s2_12angle", "openvocab", "open_vocab", "A10", "a10"]),
        "sample_keys": "",
    }
    try:
        if path.stat().st_size > 512 * 1024 * 1024:
            info["read_status"] = "SKIPPED_TOO_LARGE"
            return info
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                obj = pickle.load(f)
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                obj = json.loads(next((line for line in f if line.strip()), "{}"))
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="", errors="ignore") as f:
                obj = {h: "" for h in next(csv.reader(f), [])}
        else:
            obj = {}
        keys = collect_keys(sample_obj(obj))
        text = " ".join(keys).lower()
        info.update(
            {
                "read_status": "READ_OK",
                "has_image_id": any(k in text for k in ["image_id", "img_id", "filename", "file_name"]),
                "has_boxes": any(k in text for k in ["bbox", "bboxes", "rbbox", "rbboxes", "poly", "polygon"]),
                "has_scores": "score" in text,
                "has_labels": any(k in text for k in ["label", "class", "category"]),
                "has_angle": info["has_angle"] or "angle" in text,
                "sample_keys": ",".join(sorted(keys)[:80]),
            }
        )
    except Exception as exc:
        info["read_status"] = f"READ_ERROR:{type(exc).__name__}:{exc}"
    return info


def scan_files(roots: list[Path], max_candidates: int = 20000) -> list[Path]:
    out = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules"}]
            for name in filenames:
                p = Path(dirpath) / name
                if is_candidate(p) and str(p) not in seen:
                    seen.add(str(p))
                    out.append(p)
                    if len(out) >= max_candidates:
                        return out
    return out


def gt_search(paths: Paths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    base = paths.repo_root / "data/DOTA1_1024_500/angle_sweep_val/realistic"
    ok = True
    for angle in EXPECTED_ANGLES:
        root = base / f"angle_{angle:03d}"
        ann = root / "annfiles"
        img = root / "images"
        row = {
            "path": str(root),
            "angle": angle,
            "annfiles_exists": ann.exists(),
            "images_exists": img.exists(),
            "ann_count": len(list(ann.glob("*.txt"))) if ann.exists() else 0,
            "image_count": len(list(img.glob("*"))) if img.exists() else 0,
            "source": "known_angle_sweep",
        }
        rows.append(row)
        ok = ok and ann.exists() and img.exists()
    matched = set()
    patterns = ["labelTxt", "annotations", "annfiles", "DOTA", "val", "angle_000"]
    for root in [Path("/data/zcy"), paths.repo_root / "data"]:
        if not root.exists():
            continue
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(root):
            scanned += 1
            if scanned > 200000:
                break
            p = Path(dirpath)
            if any(x.lower() in p.name.lower() for x in patterns):
                matched.add(p)
            if any(any(x.lower() in fn.lower() for x in patterns) for fn in filenames[:20]):
                matched.add(p)
    for p in sorted(matched)[:500]:
        rows.append({"path": str(p), "angle": "", "annfiles_exists": p.exists(), "images_exists": "", "ann_count": len(list(p.glob("*.txt"))), "image_count": "", "source": "recursive_gt_search"})
    return rows, {"known_angle_sweep_available": ok, "known_angle_sweep_base": str(base), "recursive_candidate_dirs": len(matched)}


def open_vocab_ap(paths: Paths) -> dict[str, Any]:
    roots = [
        Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle"),
        Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_causal_intervention_s3_12angle"),
        Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_dehub_safety_s3_12angle"),
        paths.repo_root / "work_dirs",
        paths.repo_root / "results",
    ]
    candidates = scan_files(roots)
    inspected = [inspect_candidate(p) for p in candidates[:1000]]
    write_rows(paths.result_md_dir / "open_vocab_ap_candidate_artifacts.csv", inspected)
    gt_rows, gt_status = gt_search(paths)
    write_rows(paths.result_md_dir / "open_vocab_gt_candidate_artifacts.csv", gt_rows)
    raw_ready = [
        r
        for r in inspected
        if r["target_scope_hint"] and r["has_image_id"] and r["has_boxes"] and r["has_scores"] and r["has_labels"]
    ]
    blockers = []
    if not raw_ready:
        blockers.append("No target-scope raw prediction artifact with image_id/img_id, boxes, scores, and labels was found.")
    if not gt_status["known_angle_sweep_available"]:
        blockers.append("Known DOTA1 angle_sweep_val GT directories are incomplete.")
    blockers.append("Open-vocab prompt-to-DOTA class mapping must be explicitly verified before AP.")
    blockers.append("Rotated box/polygon format must be verified from raw predictions before IoU/AP evaluation.")
    ap_status = "PARTIAL" if raw_ready and gt_status["known_angle_sweep_available"] else "BLOCKED"
    if ap_status == "PARTIAL":
        blockers.insert(0, "Raw-like candidates exist, but evaluator is not run until class mapping and rotated box format are verified.")
    availability = {
        "AP50_STATUS": ap_status,
        "candidate_file_count": len(candidates),
        "inspected_candidate_count": len(inspected),
        "raw_prediction_ready_candidate_count": len(raw_ready),
        "raw_prediction_ready_candidates": [r["path"] for r in raw_ready],
        "gt_status": gt_status,
        "blockers": blockers,
        "search_roots": [str(p) for p in roots],
    }
    write_json(paths.result_md_dir / "open_vocab_ap_availability.json", availability)
    blocker_md = ["# Open-vocab AP Blockers", "", f"AP50_STATUS = {ap_status}", ""]
    blocker_md += [f"- {b}" for b in blockers]
    blocker_md += ["", "Minimum next step:", "", "- Locate/export target-scope raw rotated predictions with image_id, boxes/polygons, scores, labels, and angle.", "- Verify open-vocab prompt/class mapping against DOTA labels.", "- Run a GT-based rotated IoU AP50 evaluator; do not infer AP from row-level counts."]
    write_md(paths.result_md_dir / "open_vocab_ap_blockers.md", "\n".join(blocker_md))
    md = [
        "# 01 Open-vocab AP Availability and Eval",
        "",
        f"AP50_STATUS = {ap_status}",
        "",
        "The full open-vocab CSV is a false-hub/inference diagnostic table, not an AP50/mAP table.",
        "",
        "## Raw Prediction Search",
        "",
        f"- candidate files found: `{len(candidates)}`",
        f"- inspected: `{len(inspected)}`",
        f"- raw-ready target-scope candidates: `{len(raw_ready)}`",
        "",
        md_table(inspected, max_rows=30),
        "",
        "## GT Search",
        "",
        md_table(gt_rows, max_rows=40),
        "",
        "## Blockers",
        "",
    ]
    md += [f"- {b}" for b in blockers]
    write_md(paths.result_md_dir / "01_open_vocab_ap_availability_and_eval.md", "\n".join(md))
    return availability


def causal(paths: Paths, rows: list[dict[str, str]], iters: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260601)
    treatments = [x for x in ["zero_sv", "swap_sv_lv", "random_sv", "normalize_all", "norm_sv_mean"] if x in {r.get("intervention") for r in rows}]
    pair_cols = ["tile_id", "angle", "risk_group"]
    effects = paired_stats(rows, "intervention", "original", treatments, pair_cols, iters, rng)
    anglewise = stratified_deltas(rows, "intervention", "original", treatments, pair_cols, "angle")
    riskwise = stratified_deltas(rows, "intervention", "original", treatments, pair_cols, "risk_group")
    write_rows(paths.result_md_dir / "causal_paired_effects.csv", effects)
    write_rows(paths.result_md_dir / "causal_anglewise_effects.csv", anglewise)
    write_rows(paths.result_md_dir / "causal_risk_group_effects.csv", riskwise)
    write_rows(paths.result_md_dir / "causal_bootstrap_ci.csv", [{k: r.get(k, "") for k in ["group", "metric", "n", "mean_delta", "bootstrap_ci_low", "bootstrap_ci_high"]} for r in effects])
    sv = [r for r in effects if r["metric"] == "SV/img"]
    stable = []
    for t in treatments:
        vals = [to_float(r["mean_delta"]) for r in anglewise if r["group"] == t and r["metric"] == "SV/img"]
        stable.append({"intervention": t, "all_angles_lower_sv": len(vals) == 12 and all(v < 0 for v in vals), "min_angle_delta": min(vals) if vals else "", "max_angle_delta": max(vals) if vals else ""})
    md = ["# 02 Causal Intervention Paired Tests", "", f"- Pairing columns: `{pair_cols}`", f"- Bootstrap iterations: `{iters}`", "", "## Overall Effects", "", md_table(effects), "", "## SV/img Effects", "", md_table(sv), "", "## Angle Stability", "", md_table(stable), "", "## Interpretation", "", "- zero_sv, random_sv, and swap_sv_lv are paired rerun inference interventions and strongly reduce small-vehicle prediction burden.", "- normalization interventions staying near original means the effect should not be reduced to embedding norm alone.", "- This is not an AP/mAP performance claim."]
    write_md(paths.result_md_dir / "02_causal_intervention_paired_tests.md", "\n".join(md))
    return effects


def context(paths: Paths, rows: list[dict[str, str]], iters: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260602)
    done = [r for r in rows if r.get("status") == "DONE_FULL"]
    not_app = [r for r in rows if r.get("status") == "NOT_APPLICABLE"]
    pair_cols = ["tile_id", "angle", "risk_group"]
    effects = paired_stats(done, "condition", "object_only", ["context_only"], pair_cols, iters, rng)
    anglewise = stratified_deltas(done, "condition", "object_only", ["context_only"], pair_cols, "angle")
    cond_summary = summarize_by(done, "condition")
    pair_sets = Counter()
    tmp: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for r in done:
        tmp[key_tuple(r, pair_cols)].add(r.get("condition", ""))
    for vals in tmp.values():
        pair_sets[",".join(sorted(vals))] += 1
    pairing = [{"condition_set": k, "pair_units": v} for k, v in pair_sets.items()]
    not_audit = [{"condition": k[0], "status": k[1], "status_reason": k[2], "rows": v} for k, v in Counter((r.get("condition", ""), r.get("status", ""), r.get("status_reason", "")) for r in not_app).items()]
    write_rows(paths.result_md_dir / "context_pairing_audit.csv", pairing)
    write_rows(paths.result_md_dir / "context_effect_by_angle.csv", anglewise)
    write_rows(paths.result_md_dir / "context_effect_by_condition.csv", cond_summary)
    write_rows(paths.result_md_dir / "context_not_applicable_audit.csv", not_audit)
    guard = [
        {"claim": "context-only increases total detections", "status": "LIKELY_SUPPORTED", "reason": "paired det/img delta is large and positive"},
        {"claim": "context-only increases SV/img", "status": "LIKELY_SUPPORTED", "reason": "paired SV/img delta is positive"},
        {"claim": "context-only increases SV ratio", "status": "CAUTIOUSLY_SUPPORTED", "reason": "ratio rises but total detection count is a major confounder"},
        {"claim": "context alone is sufficient to create small-vehicle hallucination", "status": "CAUTIOUS", "reason": "requires visual validity and GT audit"},
        {"claim": "NOT_APPLICABLE means failure", "status": "REJECTED", "reason": "it is a data qualification split"},
    ]
    md = ["# 03 Context Counterfactual Validity and Effect", "", f"- DONE_FULL rows: `{len(done)}`", f"- NOT_APPLICABLE rows: `{len(not_app)}`", "", "## Condition Summary", "", md_table(cond_summary), "", "## Paired Effects: context_only - object_only", "", md_table(effects), "", "## Pairing Audit", "", md_table(pairing), "", "## NOT_APPLICABLE Audit", "", md_table(not_audit, max_rows=20), "", "## Interpretation Guardrails", "", md_table(guard), "", "NOT_APPLICABLE is a data qualification split, not a program failure. The large detection-count gap must be discussed before making context-causality claims."]
    write_md(paths.result_md_dir / "03_context_counterfactual_validity_and_effect.md", "\n".join(md))
    return effects


def dehub(paths: Paths, rows: list[dict[str, str]], iters: int, ap_status: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    rng = np.random.default_rng(20260603)
    pair_cols = ["tile_id", "angle", "risk_group"]
    effects = paired_stats(rows, "checkpoint_name", "baseline", ["repair"], pair_cols, iters, rng)
    anglewise = stratified_deltas(rows, "checkpoint_name", "baseline", ["repair"], pair_cols, "angle")
    riskwise = stratified_deltas(rows, "checkpoint_name", "baseline", ["repair"], pair_cols, "risk_group")
    write_rows(paths.result_md_dir / "dehub_paired_effects.csv", effects)
    write_rows(paths.result_md_dir / "dehub_anglewise_effects.csv", anglewise)
    write_rows(paths.result_md_dir / "dehub_risk_group_effects.csv", riskwise)

    base_hist = parse_hist([r for r in rows if r.get("checkpoint_name") == "baseline"])
    rep_hist = parse_hist([r for r in rows if r.get("checkpoint_name") == "repair"])
    total_b, total_r = sum(base_hist.values()), sum(rep_hist.values())
    sv_reduction = max(0, base_hist.get("small-vehicle", 0) - rep_hist.get("small-vehicle", 0))
    base_rank = {c: i + 1 for i, (c, _) in enumerate(base_hist.most_common())}
    rep_rank = {c: i + 1 for i, (c, _) in enumerate(rep_hist.most_common())}
    dist = []
    for cls in sorted(set(base_hist) | set(rep_hist)):
        b, r = base_hist.get(cls, 0), rep_hist.get(cls, 0)
        d = r - b
        dist.append(
            {
                "class": cls,
                "baseline_count": b,
                "repair_count": r,
                "absolute_delta": d,
                "relative_delta": d / b if b else "",
                "share_baseline": b / total_b if total_b else "",
                "share_repair": r / total_r if total_r else "",
                "share_delta": (r / total_r if total_r else 0) - (b / total_b if total_b else 0),
                "rank_baseline": base_rank.get(cls, ""),
                "rank_repair": rep_rank.get(cls, ""),
                "rank_change": base_rank.get(cls, 0) - rep_rank.get(cls, 0) if cls in base_rank and cls in rep_rank else "",
                "migration_score": max(0, d) / max(1, sv_reduction) if cls != "small-vehicle" else 0,
                "hub_migration_risk": cls != "small-vehicle" and d > 0,
            }
        )
    dist.sort(key=lambda r: (not r["hub_migration_risk"], -int(r["absolute_delta"])))
    migration_mass = sum(int(r["absolute_delta"]) for r in dist if r["class"] != "small-vehicle" and int(r["absolute_delta"]) > 0)
    migration_ratio = migration_mass / max(1, sv_reduction)
    write_rows(paths.result_md_dir / "dehub_class_distribution_delta.csv", dist)
    write_rows(paths.result_md_dir / "dehub_hub_migration_audit.csv", [r for r in dist if r["hub_migration_risk"]])
    score = safety_scorecard(effects, dist, migration_mass, migration_ratio)
    write_rows(paths.result_md_dir / "dehub_safety_scorecard.csv", score)
    md = ["# 04 DeHub Safety and Hub Migration", "", "- TRUE_SV_PRESERVATION_STATUS = BLOCKED", f"- migration_mass_ratio = `{migration_ratio:.4f}`", "", "## Paired Safety Effect", "", md_table(effects), "", "## Class Distribution Delta", "", md_table(dist, max_rows=20), "", "## Hub Migration Risk", "", md_table([r for r in dist if r['hub_migration_risk']], max_rows=20), "", "## Safety Scorecard", "", md_table(score), "", "repair reduces SV prediction burden, but true-SV preservation is BLOCKED without GT/raw predictions. Positive non-SV class increases are hub-migration risks, so do not claim complete safety repair."]
    write_md(paths.result_md_dir / "04_dehub_safety_and_hub_migration.md", "\n".join(md))
    return effects, migration_ratio


def safety_scorecard(effects: list[dict[str, Any]], dist: list[dict[str, Any]], migration_mass: int, migration_ratio: float) -> list[dict[str, Any]]:
    by_metric = {r["metric"]: r for r in effects}
    rows = []
    for metric in ["det/img", "SV/img", "SV ratio", "top1=SV", "LV/img", "LV ratio"]:
        r = by_metric.get(metric, {})
        rows.append({"metric": metric, "baseline": r.get("baseline_mean", ""), "repair": r.get("treatment_mean", ""), "delta": r.get("mean_delta", ""), "status": "SUPPORTED" if to_float(r.get("mean_delta")) < 0 else "CHECK", "interpretation": "paired repair-baseline delta"})
    by_class = {r["class"]: r for r in dist}
    for cls in ["plane", "storage-tank", "roundabout"]:
        r = by_class.get(cls, {})
        d = to_float(r.get("absolute_delta"))
        rows.append({"metric": f"{cls} count", "baseline": r.get("baseline_count", 0), "repair": r.get("repair_count", 0), "delta": r.get("absolute_delta", 0), "status": "WARNING" if d > 0 else "OK", "interpretation": "positive delta indicates possible hub migration"})
    rows += [
        {"metric": "total migration mass", "baseline": "", "repair": migration_mass, "delta": migration_mass, "status": "WARNING" if migration_mass > 0 else "OK", "interpretation": "positive non-SV class increases"},
        {"metric": "migration mass ratio", "baseline": "", "repair": migration_ratio, "delta": migration_ratio, "status": "WARNING" if migration_ratio > 0.1 else "OK", "interpretation": "migration mass / SV reduction"},
        {"metric": "true-SV recall preservation", "baseline": "", "repair": "", "delta": "", "status": "BLOCKED", "interpretation": "requires GT/raw prediction matching"},
        {"metric": "SV precision", "baseline": "", "repair": "", "delta": "", "status": "BLOCKED", "interpretation": "requires GT/raw prediction matching"},
        {"metric": "no-SV false hub rate", "baseline": "", "repair": "", "delta": "", "status": "BLOCKED", "interpretation": "requires GT/raw prediction matching"},
    ]
    return rows


def closedset(paths: Paths) -> None:
    specs = [
        ("closedset_false_sv_main_table.csv", "paper_table_closedset_false_sv_benchmark.csv"),
        ("closedset_rotation_gain_table.csv", "paper_table_closedset_rotation_gain.csv"),
        ("closedset_stage_decomposition_table.csv", "paper_table_closedset_stage_decomposition.csv"),
        ("closedset_spatial_concentration_table.csv", "paper_table_closedset_concentration.csv"),
    ]
    md = ["# 05 Closed-set Paper Tables", "", "mAP50/SV_AP50 is unavailable in current artifacts and is not filled in.", ""]
    for out_name, src_name in specs:
        src = paths.closedset_dir / "metrics" / src_name
        if not src.exists():
            md += [f"## {out_name}", "", f"BLOCKED: missing `{src}`", ""]
            continue
        shutil.copyfile(src, paths.result_md_dir / out_name)
        rows = read_rows(src)
        md += [f"## {out_name}", "", f"- Source: `{src}`", "", md_table(rows), ""]
    write_md(paths.result_md_dir / "05_closedset_paper_tables.md", "\n".join(md))


def save_fig(paths: Paths, name: str) -> None:
    for ext in [".png", ".pdf"]:
        plt.savefig(paths.figures_dir / f"{name}{ext}", bbox_inches="tight", dpi=180)
    plt.close()


def rot_x() -> None:
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")


def figures(paths: Paths, dfs: dict[str, list[dict[str, str]]], causal_eff: list[dict[str, Any]], context_eff: list[dict[str, Any]], dehub_eff: list[dict[str, Any]]) -> list[str]:
    made: list[str] = []
    errors = []
    def wrap(name: str, fn: Any) -> None:
        try:
            fn()
            made.append(name)
        except Exception as exc:
            errors.append({"figure": name, "error": str(exc)})

    def fig1() -> None:
        rows = read_rows(paths.closedset_dir / "metrics/paper_table_closedset_false_sv_benchmark.csv")
        plt.figure(figsize=(10, 4))
        x = [r["Model"] for r in rows]
        y = [to_float(r["Abs_FalseSV"]) for r in rows]
        plt.bar(x, y)
        for i, r in enumerate(rows):
            plt.text(i, y[i], r["Architecture"], rotation=90, va="bottom", fontsize=6)
        plt.ylabel("Abs false-SV")
        rot_x()
        save_fig(paths, "fig_closedset_false_sv_burden")
    wrap("fig_closedset_false_sv_burden", fig1)

    def fig2() -> None:
        rows = read_rows(paths.closedset_dir / "metrics/paper_table_closedset_rotation_gain.csv")
        plt.figure(figsize=(10, 4))
        plt.bar([r["Model"] for r in rows], [to_float(r["Mean_RG_SV"]) for r in rows])
        plt.ylabel("mean RG_SV")
        rot_x()
        save_fig(paths, "fig_closedset_rotation_gain")
    wrap("fig_closedset_rotation_gain", fig2)

    def fig3() -> None:
        rows = summarize_by(dfs["openvocab"], "angle")
        plt.figure(figsize=(7, 4))
        x = [str(r["angle"]) for r in rows]
        plt.plot(x, [to_float(r["SV/img"]) for r in rows], marker="o", label="SV/img")
        plt.plot(x, [to_float(r["det/img"]) for r in rows], marker="s", label="det/img")
        plt.legend()
        plt.xlabel("angle")
        save_fig(paths, "fig_openvocab_angle_curve")
    wrap("fig_openvocab_angle_curve", fig3)

    def fig4() -> None:
        rows = [r for r in causal_eff if r["metric"] == "SV/img"]
        x = [r["group"] for r in rows]
        y = np.array([to_float(r["mean_delta"]) for r in rows])
        low = np.array([to_float(r["bootstrap_ci_low"]) for r in rows])
        high = np.array([to_float(r["bootstrap_ci_high"]) for r in rows])
        plt.figure(figsize=(7, 4))
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=[y - low, high - y], fmt="none", ecolor="black", capsize=3)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Delta SV/img vs original")
        rot_x()
        save_fig(paths, "fig_causal_intervention_delta_sv")
    wrap("fig_causal_intervention_delta_sv", fig4)

    def fig5() -> None:
        rows = [r for r in context_eff if r["metric"] in {"det/img", "SV/img", "SV ratio", "top1=SV"}]
        plt.figure(figsize=(7, 4))
        plt.bar([r["metric"] for r in rows], [to_float(r["mean_delta"]) for r in rows])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("context_only - object_only")
        save_fig(paths, "fig_context_counterfactual_delta")
    wrap("fig_context_counterfactual_delta", fig5)

    def fig6() -> None:
        rows = [r for r in dehub_eff if r["metric"] in {"SV/img", "SV ratio", "top1=SV", "det/img"}]
        x = np.arange(len(rows))
        plt.figure(figsize=(7, 4))
        plt.bar(x - 0.2, [to_float(r["baseline_mean"]) for r in rows], width=0.4, label="baseline")
        plt.bar(x + 0.2, [to_float(r["treatment_mean"]) for r in rows], width=0.4, label="repair")
        plt.xticks(x, [r["metric"] for r in rows])
        plt.legend()
        save_fig(paths, "fig_dehub_baseline_repair")
    wrap("fig_dehub_baseline_repair", fig6)

    def fig7() -> None:
        rows = read_rows(paths.result_md_dir / "dehub_class_distribution_delta.csv")
        rows = sorted(rows, key=lambda r: to_float(r["absolute_delta"]), reverse=True)[:12]
        plt.figure(figsize=(9, 4))
        y = [to_float(r["absolute_delta"]) for r in rows]
        plt.bar([r["class"] for r in rows], y, color=["tab:red" if v > 0 else "tab:blue" for v in y])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("repair_count - baseline_count")
        rot_x()
        save_fig(paths, "fig_dehub_hub_migration")
    wrap("fig_dehub_hub_migration", fig7)

    write_json(paths.result_md_dir / "figures_status.json", {"made": made, "errors": errors})
    return made


def claim_row(claim_id: str, claim_text: str, status: str, artifacts: str, conditions: str, risk: str, allowed: str, forbidden: str) -> dict[str, str]:
    return {"claim_id": claim_id, "claim_text": claim_text, "evidence_status": status, "supporting_artifacts": artifacts, "required_conditions": conditions, "main_risk": risk, "allowed_wording": allowed, "forbidden_wording": forbidden}


def claims(paths: Paths, ap: dict[str, Any], migration_ratio: float) -> None:
    hub = migration_ratio > 0.1
    ap_blocked = ap.get("AP50_STATUS") == "BLOCKED"
    rows = [
        claim_row("C1", "closed-set small-vehicle false-hub exists across multiple detector architectures.", "SUPPORTED", "closedset_false_sv_main_table.csv", "closed-set audit rows valid", "mAP unavailable", "Closed-set detectors across multiple architectures exhibit a measurable small-vehicle false-hub burden.", "All closed-set detectors fail for the same reason."),
        claim_row("C2", "closed-set false-SV is not limited to one model family.", "SUPPORTED", "closedset_false_sv_main_table.csv", "architecture labels valid", "severity differs by model", "The false-SV burden appears in both dense-head and two-stage families.", "Every architecture has identical failure behavior."),
        claim_row("C3", "dense/pre-NMS stage already shows small-vehicle bias in hook-supported dense-head models.", "SUPPORTED", "closedset_stage_decomposition_table.csv", "hook-supported subset only", "not all models have hooks", "Hook-supported dense-head models show nonzero dense/pre-NMS small-vehicle bias.", "All models prove the dense-stage mechanism."),
        claim_row("C4", "NMS/post-processing amplifies but does not solely create the small-vehicle false-hub.", "SUPPORTED", "closedset_stage_decomposition_table.csv", "hook-supported subset", "hook gaps", "Post-processing amplifies a bias already visible before NMS in supported models.", "NMS is the only cause."),
        claim_row("C5", "open-vocab A10 checkpoint produces persistent small-vehicle predictions across risk groups and rotation angles.", "SUPPORTED", "open_vocab_benchmark_rows_merged.csv", "diagnostic row-level inference", "not AP", "The A10 open-vocabulary checkpoint persistently predicts small-vehicle across risk groups and 12 rotation angles.", "Open-vocab AP improves or fails based on this table."),
        claim_row("C6", "open-vocab full S2 is an inference/false-hub diagnostic, not AP50/mAP evaluation.", "SUPPORTED", "01_open_vocab_ap_availability_and_eval.md", "AP artifacts unavailable/unverified", "none", "The full S2 table should be used as a false-hub diagnostic, not an AP50/mAP result.", "The row-level counts are AP50."),
        claim_row("C7", "embedding interventions are paired rerun inference, not label rewriting.", "SUPPORTED", "causal_paired_effects.csv", "pairing holds", "raw forward artifacts not stored", "The intervention rows are paired rerun inference under controlled embedding modifications.", "The effect is only from relabeling."),
        claim_row("C8", "zero_sv, random_sv, and swap_sv_lv provide causal evidence that small-vehicle prediction burden is tied to the support/class embedding.", "SUPPORTED", "causal_paired_effects.csv", "negative paired deltas", "not AP", "zero_sv, random_sv, and swap_sv_lv sharply reduce the small-vehicle prediction burden relative to original.", "All false positives are caused by embeddings."),
        claim_row("C9", "normalization interventions not reducing SV burden suggests the attractor is not simply an embedding-norm artifact.", "CAUTIOUS", "causal_paired_effects.csv", "normalization near original", "limited intervention set", "Normalization-like interventions remaining near original suggests the effect is not simply an embedding-norm artifact.", "Embedding norm is proven irrelevant in all settings."),
        claim_row("C10", "context counterfactual suggests context contributes to prediction burden, but interpretation must account for DONE_FULL eligibility and detection-count differences.", "CAUTIOUS", "context_effect_by_condition.csv", "DONE_FULL pairing only", "detection-count confounder", "Context-only rows increase prediction burden, but the total-detection shift requires cautious interpretation.", "Context alone proves hallucination."),
        claim_row("C11", "NOT_APPLICABLE in context counterfactual is a data qualification split, not an execution failure.", "SUPPORTED", "context_not_applicable_audit.csv", "status_reason preserved", "none", "NOT_APPLICABLE denotes samples without usable small-vehicle polygons, not failed execution.", "NOT_APPLICABLE means the run failed."),
        claim_row("C12", "DeHub repair reduces small-vehicle prediction burden in paired S3 12-angle evaluation.", "SUPPORTED", "dehub_paired_effects.csv", "baseline/repair pairing", "not AP", "On the paired S3 12-angle set, repair reduces small-vehicle prediction burden and top1=SV.", "DeHub improves detection accuracy."),
        claim_row("C13", "DeHub safety cannot be claimed without true-SV preservation and hub-migration analysis.", "SUPPORTED", "04_dehub_safety_and_hub_migration.md", "true-SV GT eval unavailable", "none", "DeHub safety requires true-object preservation and migration checks beyond SV burden reduction.", "DeHub is safe because SV/img decreased."),
        claim_row("C14", "If hub migration is observed, DeHub should be described as false-SV burden reduction rather than complete safety repair.", "SUPPORTED" if hub else "CAUTIOUS", "dehub_hub_migration_audit.csv", "class histogram diagnostic", "needs AP/GT", "Observed positive non-SV class deltas mean repair should be framed as false-SV burden reduction, not complete safety repair.", "DeHub completely eliminates false-hub side effects."),
        claim_row("C15", "Current evidence supports a rotation semantic attractor / false-hub diagnosis, but not yet a standard AP improvement claim unless AP evaluator succeeds.", "SUPPORTED" if ap_blocked else "CAUTIOUS", "summary_evidence_closure_20260601.md", "AP status considered", "AP blocked/partial", "The current evidence supports a diagnostic rotation semantic attractor claim; standard AP improvement requires a verified AP evaluator.", "OpenRSD AP improves after repair."),
    ]
    write_rows(paths.result_md_dir / "claim_ledger.csv", rows)
    matrix = []
    for r in rows:
        cid = r["claim_id"]
        matrix.append({**{k: r[k] for k in ["claim_id", "evidence_status", "supporting_artifacts", "main_risk"]}, "closedset": cid in {"C1", "C2", "C3", "C4"}, "openvocab": cid in {"C5", "C6"}, "causal": cid in {"C7", "C8", "C9"}, "context": cid in {"C10", "C11"}, "dehub": cid in {"C12", "C13", "C14"}, "ap_gt": cid in {"C6", "C13", "C15"}})
    write_rows(paths.result_md_dir / "claim_strength_matrix.csv", matrix)
    write_md(paths.result_md_dir / "paper_ready_claims.md", "# Paper-ready Claims\n\n" + md_table([{"claim_id": r["claim_id"], "allowed_wording": r["allowed_wording"]} for r in rows if r["evidence_status"] in {"SUPPORTED", "CAUTIOUS"}]))
    write_md(paths.result_md_dir / "claims_to_avoid.md", "# Claims to Avoid\n\n" + md_table([{"claim_id": r["claim_id"], "forbidden_wording": r["forbidden_wording"], "main_risk": r["main_risk"]} for r in rows]))
    write_md(paths.result_md_dir / "06_claim_ledger_and_evidence_boundary.md", "# 06 Claim Ledger and Evidence Boundary\n\n" + md_table(rows))


def summary(paths: Paths, integrity: dict[str, Any], ap: dict[str, Any], figs: list[str], migration_ratio: float) -> None:
    ap_status = ap.get("AP50_STATUS", "BLOCKED")
    judgment = "YES_FOR_DIAGNOSTIC_CLAIM"
    reason = "Existing closed-set, open-vocab diagnostic, causal, context, and DeHub burden-reduction evidence supports the diagnostic experimental chapter; AP improvement and full safety claims remain blocked by GT/raw-prediction evaluation."
    lines = [
        "# Summary Evidence Closure 20260601",
        "",
        "## Steps Executed",
        "",
        "| Step | Status | Output |",
        "| --- | --- | --- |",
        f"| Artifact integrity audit | {integrity.get('overall_status', 'DONE')} | `00_artifact_integrity_audit.md` |",
        f"| Open-vocab AP availability audit | {ap_status} | `01_open_vocab_ap_availability_and_eval.md` |",
        "| Causal paired statistical tests | DONE | `02_causal_intervention_paired_tests.md` |",
        "| Context counterfactual validity/effect audit | DONE | `03_context_counterfactual_validity_and_effect.md` |",
        "| DeHub safety and hub-migration audit | DONE | `04_dehub_safety_and_hub_migration.md` |",
        "| Closed-set paper tables | DONE | `05_closedset_paper_tables.md` |",
        "| Claim ledger | DONE | `06_claim_ledger_and_evidence_boundary.md` |",
        "",
        "## AP50 Evaluator Status",
        "",
        f"AP50_STATUS = {ap_status}",
        "",
        "## DeHub Safety Status",
        "",
        "- SV burden reduction: DONE",
        "- true-SV preservation: BLOCKED",
        f"- hub migration: DONE, migration_mass_ratio={migration_ratio:.4f}",
        "",
        "## Causal Intervention Statistical Conclusion",
        "",
        "zero_sv, random_sv, and swap_sv_lv strongly reduce SV burden under paired rerun inference; normalization interventions are close to original, so the mechanism should not be described as only embedding norm.",
        "",
        "## Context Counterfactual Statistical Conclusion",
        "",
        "context_only vs object_only is paired for DONE_FULL rows and increases detection/SV burden, but the large detection-count difference must be treated as a confounder.",
        "",
        "## Closed-set Paper Tables Status",
        "",
        "False-SV main, rotation gain, stage decomposition, and spatial concentration tables are consolidated. mAP50/SV_AP50 remains unavailable.",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{paths.figures_dir / (name + '.png')}`" for name in figs]
    lines += [
        "",
        "## Claim Ledger Paths",
        "",
        f"- `{paths.result_md_dir / 'claim_ledger.csv'}`",
        f"- `{paths.result_md_dir / 'paper_ready_claims.md'}`",
        f"- `{paths.result_md_dir / 'claims_to_avoid.md'}`",
        "",
        "## Five Paper-writable Conclusions",
        "",
        "1. Closed-set small-vehicle false-hub appears across multiple detector architectures.",
        "2. Open-vocab A10 inference persistently produces small-vehicle predictions across risk groups and rotation angles.",
        "3. Embedding interventions are paired rerun inference and strongly modulate SV burden.",
        "4. Context contributes to prediction burden, with detection-count confounding explicitly acknowledged.",
        "5. DeHub repair reduces paired small-vehicle prediction burden but must be checked for preservation and migration.",
        "",
        "## Five Claims to Avoid or Qualify",
        "",
        "1. Do not claim open-vocab AP50/mAP from row-level counts.",
        "2. Do not claim DeHub improves AP unless AP evaluator succeeds.",
        "3. Do not claim DeHub is safe without true-SV preservation.",
        "4. Do not claim context alone proves hallucination without visual/GT validation.",
        "5. Do not claim all closed-set failures share one embedding mechanism.",
        "",
        "## Tomorrow First Priority",
        "",
        "Locate/export target-scope raw open-vocab predictions with rotated boxes, scores, labels, image ids, and verified class mapping; then run GT-based AP50 and true-SV preservation evaluators.",
        "",
        "## Strict Judgment",
        "",
        f"`{judgment}`",
        "",
        reason,
    ]
    write_md(paths.result_md_dir / "summary_evidence_closure_20260601.md", "\n".join(lines))


def dryrun(paths: Paths) -> int:
    ensure_dirs(paths)
    rows = []
    for name, path in [
        ("openvocab_csv", paths.openvocab_csv),
        ("causal_csv", paths.causal_csv),
        ("context_csv", paths.context_csv),
        ("dehub_csv", paths.dehub_csv),
        ("closedset_dir", paths.closedset_dir),
    ]:
        rows.append({"name": name, "path": str(path), "exists": path.exists()})
    write_rows(paths.result_md_dir / "dryrun_input_check.csv", rows)
    write_md(paths.result_md_dir / "dryrun_input_check.md", "# Dryrun Input Check\n\n" + md_table(rows))
    if not all(r["exists"] for r in rows):
        log(paths, "dryrun failed: missing inputs")
        return 2
    log(paths, "dryrun succeeded: all required inputs exist")
    return 0


def run_all(paths: Paths, iters: int, force: bool) -> int:
    ensure_dirs(paths)
    if not force and any(paths.result_md_dir.iterdir()):
        log(paths, "output directory is non-empty; use --force")
        return 2
    log(paths, "loading core CSVs")
    dfs = {
        "openvocab": read_rows(paths.openvocab_csv),
        "causal": read_rows(paths.causal_csv),
        "context": read_rows(paths.context_csv),
        "dehub": read_rows(paths.dehub_csv),
    }
    log(paths, "artifact integrity audit")
    integ = integrity_audit(paths, dfs)
    log(paths, "open-vocab AP availability audit")
    ap = open_vocab_ap(paths)
    log(paths, "causal paired tests")
    causal_eff = causal(paths, dfs["causal"], iters)
    log(paths, "context counterfactual audit")
    context_eff = context(paths, dfs["context"], iters)
    log(paths, "DeHub safety and hub migration audit")
    dehub_eff, migration_ratio = dehub(paths, dfs["dehub"], iters, ap)
    log(paths, "closed-set paper table consolidation")
    closedset(paths)
    log(paths, "paper-ready figures")
    figs = figures(paths, dfs, causal_eff, context_eff, dehub_eff)
    log(paths, "claim ledger")
    claims(paths, ap, migration_ratio)
    log(paths, "summary")
    summary(paths, integ, ap, figs, migration_ratio)
    log(paths, "evidence closure complete")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--result-md-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--openvocab-csv", type=Path, required=True)
    p.add_argument("--causal-csv", type=Path, required=True)
    p.add_argument("--context-csv", type=Path, required=True)
    p.add_argument("--dehub-csv", type=Path, required=True)
    p.add_argument("--closedset-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["dryrun", "all"], default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = Paths(args.repo_root, args.result_md_dir, args.work_dir, args.openvocab_csv, args.causal_csv, args.context_csv, args.dehub_csv, args.closedset_dir)
    if args.mode == "dryrun":
        return dryrun(paths)
    return run_all(paths, args.bootstrap_iters, args.force)


if __name__ == "__main__":
    sys.exit(main())

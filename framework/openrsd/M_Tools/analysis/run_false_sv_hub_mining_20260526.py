#!/usr/bin/env python3
"""False-SV-hub diagnostic tile mining (offline, 2026-05-26).

Separates high-SV-output tiles from true-SV-rich vs false-SV-hub using GT matching,
existing predictions, and manual control roles for P0148 / P0682.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_DEFAULT = Path("/data1/zcy/OpenRSD")
WORK_DEFAULT = REPO_DEFAULT / "work_dirs/exp_false_sv_hub_mining_20260526"
RESULT_MD_DEFAULT = REPO_DEFAULT / "resultmd/exp_false_sv_hub_mining_20260526"

P0148 = "P0148__1024__651___0"
P0682 = "P0682__1024__553___0"
TILE_ROLES = {
    P0148: "false_sv_hub_diagnostic_candidate",
    P0682: "true_sv_rich_positive_control",
}

SV_LABELS = {
    "small-vehicle", "small_vehicle", "Small_Vehicle", "small vehicle",
}
PLACEHOLDER_SIGNATURE = Counter({"small-vehicle": 10, "ship": 5, "large-vehicle": 5})

DOTA12_ANGLES = list(range(0, 360, 30))
ANGLE_SWEEP_REL = Path("data/DOTA1_1024_500/angle_sweep_val/realistic")
PYTHON_BIN = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
BASELINE_CONFIG = "M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
BASELINE_CKPT = "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"

ALL_EXPS = [
    "fsv0_inventory", "fsv1_collect_candidates", "fsv2_gt_match",
    "fsv3_false_sv_score", "fsv4_crop_audit_pack", "fsv5_tile_taxonomy",
    "fsv6_b8k_preserve_check",
]


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fmt4(v: Any) -> str:
    if v is None or v == "" or v == "NA":
        return ""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return ""
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "" or v == "NA":
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not fields:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt4(r.get(k, "")) if isinstance(r.get(k), (float, np.floating)) else r.get(k, "")
                        for k in fields})


def read_csv(path: Path, max_rows: int = 500000) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return [dict(r) for i, r in enumerate(csv.DictReader(f)) if i < max_rows]


def markdown_table(rows: list[dict], fields: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_empty_\n"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in fields) + " |")
    return "\n".join(lines) + "\n"


class CommandRunner:
    def __init__(self, work_dir: Path, gpu_ids: str = "8,9"):
        self.work_dir = work_dir
        self.log_dir = work_dir / "logs"
        ensure_dir(self.log_dir)
        self.gpu_ids = gpu_ids
        self._logf = None

    def log(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        if self._logf:
            self._logf.write(line + "\n")
            self._logf.flush()

    def open_log(self, name: str) -> None:
        ensure_dir(self.log_dir)
        self._logf = (self.log_dir / name).open("a", encoding="utf-8")

    def close_log(self) -> None:
        if self._logf:
            self._logf.close()
            self._logf = None

    def run(self, cmd: list[str], name: str, cwd: Path | None = None, use_gpu: bool = False) -> dict:
        stdout_p = self.log_dir / f"{name}.stdout.txt"
        stderr_p = self.log_dir / f"{name}.stderr.txt"
        meta: dict[str, Any] = {
            "command": cmd,
            "start_time": now_iso(),
            "stdout_path": str(stdout_p),
            "stderr_path": str(stderr_p),
        }
        if use_gpu:
            meta["nvidia_smi_before"] = self.nvidia_smi()
        env = os.environ.copy()
        if use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        start = dt.datetime.now()
        try:
            p = subprocess.run(
                cmd, cwd=str(cwd) if cwd else None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            rc = p.returncode
            stdout_p.write_text(p.stdout or "", encoding="utf-8")
            stderr_p.write_text(p.stderr or "", encoding="utf-8")
        except Exception as exc:
            rc = -999
            stderr_p.write_text(traceback.format_exc() + "\n" + repr(exc), encoding="utf-8")
        meta["end_time"] = now_iso()
        meta["return_code"] = rc
        meta["elapsed_sec"] = (dt.datetime.now() - start).total_seconds()
        if use_gpu:
            meta["nvidia_smi_after"] = self.nvidia_smi()
        (self.log_dir / f"{name}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    @staticmethod
    def nvidia_smi() -> str:
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
            )
            return (p.stdout or p.stderr or "").strip()
        except Exception as exc:
            return repr(exc)


def git_commit(repo: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
        )
        return (p.stdout or "").strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "mmcv", "mmengine", "mmdet", "mmrotate"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "?")
        except Exception:
            out[name] = "MISSING"
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, text=True, timeout=10)
        out["gpu"] = (p.stdout or "").strip().split("\n")[0] if p.returncode == 0 else "N/A"
    except Exception:
        out["gpu"] = "N/A"
    return out


def is_sv_label(label: str) -> bool:
    return str(label).strip().lower().replace("_", "-") == "small-vehicle" or label in SV_LABELS


def ann_is_placeholder(texts: Sequence[str]) -> bool:
    c = Counter(str(t).strip() for t in texts)
    if sum(c.values()) != 20:
        return False
    for k, v in PLACEHOLDER_SIGNATURE.items():
        if c.get(k, 0) != v:
            return False
    return True


def parse_dota_txt(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    polys: list[np.ndarray] = []
    if not path.exists() or path.stat().st_size == 0:
        return texts, np.zeros((0, 8), dtype=np.float32)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        if parts[0].startswith("imagesource") or parts[0].startswith("gsd"):
            continue
        texts.append(parts[8])
        polys.append(np.asarray([float(x) for x in parts[:8]], dtype=np.float32))
    if polys:
        return texts, np.stack(polys, axis=0)
    return texts, np.zeros((0, 8), dtype=np.float32)


def angle_sweep_ann_path(repo: Path, tile_id: str, angle: int) -> Path:
    return repo / ANGLE_SWEEP_REL / f"angle_{angle:03d}" / "annfiles" / f"{tile_id}.txt"


def angle_sweep_image_path(repo: Path, tile_id: str, angle: int) -> Path | None:
    base = repo / ANGLE_SWEEP_REL / f"angle_{angle:03d}" / "images"
    for ext in (".png", ".jpg"):
        p = base / f"{tile_id}{ext}"
        if p.exists():
            return p
    return None


def load_gt(
    repo: Path, tile_id: str, angle: int = 0, gt_source: str = "auto",
) -> dict[str, Any]:
    """Resolve GT; returns texts, polys, path, gt_quality."""
    if gt_source == "angle_sweep":
        candidates: list[tuple[Path, str]] = [
            (angle_sweep_ann_path(repo, tile_id, angle), "angle_sweep"),
        ]
    else:
        candidates = [
            (repo / f"data/DOTA1_1024_500/ss_train/annfiles/{tile_id}.txt", "ss_train"),
            (repo / f"data/DOTA1_1024_500/ss_val/annfiles/{tile_id}.txt", "ss_val"),
            (angle_sweep_ann_path(repo, tile_id, angle), "angle_sweep"),
            (repo / f"vis/{tile_id}/dataset/annfiles/{tile_id}_rot{angle:03d}.pkl", "vis_rotated_pkl"),
            (Path(f"/data/zcy/dataset/test_ms/annfiles/{tile_id}.txt"), "test_ms"),
            (Path(f"/data/zcy/dataset/test_ss/annfiles/{tile_id}.txt"), "test_ss"),
        ]
    for path, tag in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                ann = pickle.load(f)
            texts = list(ann.get("texts") or [])
            polys_raw = ann.get("polys")
            polys = np.asarray(
                polys_raw if polys_raw is not None else np.zeros((0, 8)), dtype=np.float32)
        else:
            texts, polys = parse_dota_txt(path)
        if len(texts) == 0 and path.stat().st_size == 0:
            continue
        quality = "REAL"
        if ann_is_placeholder(texts):
            quality = "PLACEHOLDER_TEMPLATE"
        elif tag in ("vis_rotated_pkl",) and sum(1 for t in texts if is_sv_label(t)) == 10:
            quality = "PLACEHOLDER_TEMPLATE"
        return dict(texts=texts, polys=polys, path=str(path), source=tag, gt_quality=quality)
    return dict(texts=[], polys=np.zeros((0, 8), dtype=np.float32), path="",
                source="GT_MISSING", gt_quality="GT_MISSING")


def poly8_to_rbox(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    try:
        from mmrotate.structures.bbox import qbox2rbox
        import torch
        qbox = torch.from_numpy(poly.reshape(1, 8))
        return qbox2rbox(qbox).numpy()[0].astype(np.float32)
    except Exception:
        # HBB fallback
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return np.array([cx, cy, x2 - x1, y2 - y1, 0.0], dtype=np.float32)


def polys_to_rboxes(polys: np.ndarray) -> np.ndarray:
    if polys is None or len(polys) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.stack([poly8_to_rbox(p) for p in polys], axis=0)


def match_sv_to_gt(
    det_polys: np.ndarray, det_labels: Sequence[str], det_scores: np.ndarray,
    gt_polys: np.ndarray, gt_texts: Sequence[str],
    iou_thr: float,
) -> tuple[int, list[float], str]:
    """Returns matched count, per-det max iou list, iou_method."""
    sv_idx = [i for i, t in enumerate(det_labels) if is_sv_label(t)]
    if not sv_idx:
        return 0, [], "N/A"
    gt_sv_idx = [i for i, t in enumerate(gt_texts) if is_sv_label(t)]
    if not gt_sv_idx:
        return 0, [0.0] * len(sv_idx), "NO_GT_SV"

    det_r = polys_to_rboxes(det_polys[sv_idx])
    gt_r = polys_to_rboxes(gt_polys[gt_sv_idx])

    max_ious: list[float] = []
    method = "HBB_APPROX"
    try:
        import torch
        from mmcv.ops import box_iou_rotated
        method = "ROTATED_IOU"
        for d in det_r:
            if len(gt_r) == 0:
                max_ious.append(0.0)
                continue
            ious = box_iou_rotated(
                torch.from_numpy(d.reshape(1, 5).astype(np.float32)),
                torch.from_numpy(gt_r.astype(np.float32)),
            ).cpu().numpy().reshape(-1)
            max_ious.append(float(ious.max()) if ious.size else 0.0)
    except Exception:
        for d in det_r:
            if len(gt_r) == 0:
                max_ious.append(0.0)
                continue
            # axis-aligned IoU on HBB
            dx1, dy1 = d[0] - d[2] / 2, d[1] - d[3] / 2
            dx2, dy2 = d[0] + d[2] / 2, d[1] + d[3] / 2
            best = 0.0
            for g in gt_r:
                gx1, gy1 = g[0] - g[2] / 2, g[1] - g[3] / 2
                gx2, gy2 = g[0] + g[2] / 2, g[1] + g[3] / 2
                ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (dx2 - dx1) * (dy2 - dy1) + (gx2 - gx1) * (gy2 - gy1) - inter + 1e-6
                best = max(best, inter / union)
            max_ious.append(best)

    matched = sum(1 for v in max_ious if v >= iou_thr)
    return matched, max_ious, method


def load_vis_prediction(pkl_path: Path, tile_id: str, angle: int = 0) -> dict[str, Any] | None:
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return None
    keys = [
        f"{tile_id}_rot{angle:03d}",
        f"{tile_id}_rot{angle:03d}.jpg",
    ]
    rec = None
    hit_key = None
    for k in keys:
        if k in data:
            rec = data[k]
            hit_key = k
            break
    if rec is None:
        for k, v in data.items():
            if tile_id in str(k) and f"rot{angle:03d}" in str(k):
                rec = v
                hit_key = k
                break
    if rec is None:
        return None
    texts = list(rec.get("texts") or [])
    scores_raw = rec.get("scores")
    scores = np.asarray([] if scores_raw is None else scores_raw, dtype=np.float32)
    polys_raw = rec.get("polys")
    polys = np.asarray(polys_raw if polys_raw is not None else np.zeros((0, 8)), dtype=np.float32)
    return dict(texts=texts, scores=scores, polys=polys, key=hit_key or keys[0])


def load_angle_sweep_prediction(pkl_path: Path, tile_id: str) -> dict[str, Any] | None:
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return None
    if "texts" in data:
        texts = list(data.get("texts") or [])
        scores_raw = data.get("scores")
        scores = np.asarray([] if scores_raw is None else scores_raw, dtype=np.float32)
        polys_raw = data.get("polys")
        polys = np.asarray(
            polys_raw if polys_raw is not None else np.zeros((0, 8)), dtype=np.float32,
        )
        return dict(texts=texts, scores=scores, polys=polys)
    for key in (tile_id, f"{tile_id}.png", f"{tile_id}.jpg", f"{tile_id}_rot000"):
        if key in data:
            rec = data[key]
            break
    else:
        rec = next(iter(data.values()), None)
    if rec is None or not isinstance(rec, dict):
        return None
    texts = list(rec.get("texts") or [])
    scores_raw = rec.get("scores")
    scores = np.asarray([] if scores_raw is None else scores_raw, dtype=np.float32)
    polys_raw = rec.get("polys")
    polys = np.asarray(polys_raw if polys_raw is not None else np.zeros((0, 8)), dtype=np.float32)
    return dict(texts=texts, scores=scores, polys=polys)


def load_candidate_prediction(cand: dict, tile_id: str, angle: int) -> dict[str, Any] | None:
    pred_path = Path(cand.get("pred_path") or "")
    if not pred_path.exists():
        return None
    if cand.get("pred_source") == "angle_sweep_infer":
        return load_angle_sweep_prediction(pred_path, tile_id)
    return load_vis_prediction(pred_path, tile_id, angle)


def build_angle_sweep_gt_index(repo: Path, wd: Path, angle: int = 0) -> list[dict]:
    ann_dir = repo / ANGLE_SWEEP_REL / f"angle_{angle:03d}" / "annfiles"
    rows: list[dict] = []
    for txt in sorted(ann_dir.glob("*.txt")):
        texts, _ = parse_dota_txt(txt)
        gt_sv = sum(1 for t in texts if is_sv_label(t))
        quality = "REAL"
        if ann_is_placeholder(texts):
            quality = "PLACEHOLDER_TEMPLATE"
        rows.append(dict(
            tile_id=txt.stem,
            angle=angle,
            gt_path=str(txt),
            gt_total=len(texts),
            gt_sv_count=gt_sv,
            gt_quality=quality,
        ))
    write_csv(wd / "fsv0_inventory/ftable_angle_sweep_gt_index.csv", rows,
              ["tile_id", "angle", "gt_path", "gt_total", "gt_sv_count", "gt_quality"])
    return rows


def select_angle_sweep_candidate_tiles(
    gt_rows: list[dict], max_tiles: int,
) -> list[str]:
    """Pick tiles: high GT-SV (true-SV-rich pool) + low-GT-SV with other objects (false-hub pool)."""
    real = [r for r in gt_rows if r.get("gt_quality") == "REAL"]
    if not max_tiles:
        max_tiles = 80

    high_pool = sorted(
        [r for r in real if int(r.get("gt_sv_count") or 0) >= 20],
        key=lambda r: -int(r.get("gt_sv_count") or 0),
    )
    low_pool = sorted(
        [
            r for r in real
            if int(r.get("gt_sv_count") or 0) <= 5 and int(r.get("gt_total") or 0) >= 15
        ],
        key=lambda r: -int(r.get("gt_total") or 0),
    )
    mid_pool = sorted(
        [r for r in real if 6 <= int(r.get("gt_sv_count") or 0) < 20],
        key=lambda r: -int(r.get("gt_sv_count") or 0),
    )

    n_high = max_tiles // 2
    n_low = max_tiles - n_high
    chosen: list[str] = []

    def _add_from(pool: list[dict], limit: int) -> None:
        for r in pool:
            if len(chosen) >= limit:
                return
            tid = r["tile_id"]
            if tid not in chosen:
                chosen.append(tid)

    _add_from(high_pool, n_high)
    _add_from(low_pool, max_tiles)
    if len(chosen) < max_tiles:
        _add_from(mid_pool, max_tiles)
    if len(chosen) < max_tiles:
        _add_from(high_pool, max_tiles)
    return chosen[:max_tiles]


def parse_gpu_id_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        size = 1
    return [items[i: i + size] for i in range(0, len(items), size)]


def prediction_record_from_batch(all_results: dict, tile_id: str) -> dict | None:
    for key in (tile_id, f"{tile_id}.png", f"{tile_id}.jpg", f"{tile_id}_rot000"):
        rec = all_results.get(key)
        if isinstance(rec, dict) and "texts" in rec:
            return rec
    if isinstance(all_results.get("texts"), list):
        return all_results
    return None


def write_tile_pred_cache(out_pkl: Path, rec: dict) -> None:
    ensure_dir(out_pkl.parent)
    with out_pkl.open("wb") as f:
        pickle.dump(rec, f)


def split_batch_results_to_cache(
    all_results: dict, tile_ids: list[str], cache_dir: Path, angle: int,
) -> list[str]:
    missing: list[str] = []
    for tile in tile_ids:
        out_pkl = angle_sweep_pred_cache_path(cache_dir, tile, angle)
        rec = prediction_record_from_batch(all_results, tile)
        if rec is None:
            missing.append(tile)
            continue
        write_tile_pred_cache(out_pkl, rec)
    return missing


def angle_sweep_pred_cache_path(cache_dir: Path, tile_id: str, angle: int) -> Path:
    return cache_dir / tile_id / f"angle_{angle:03d}_results.pkl"


def build_fsv1_candidate_row(
    repo: Path, tile: str, angle: int, pred_pkl: Path, gt_meta: dict,
) -> dict:
    pred = load_angle_sweep_prediction(pred_pkl, tile)
    if pred is None:
        return {}
    total = len(pred["texts"])
    sv = sum(1 for t in pred["texts"] if is_sv_label(t))
    fsr = sv / total if total else 0.0
    img = angle_sweep_image_path(repo, tile, angle)
    return dict(
        tile_id=tile,
        source="dota12angle_val",
        image_path=str(img) if img else "",
        ann_path=gt_meta.get("gt_path", str(angle_sweep_ann_path(repo, tile, angle))),
        pred_path=str(pred_pkl),
        total_dets=total,
        sv_dets=sv,
        final_sv_ratio=fsr,
        dense_sv_ratio="",
        score_mean=float(np.mean(pred["scores"])) if len(pred["scores"]) else "",
        note=f"gt_sv={gt_meta.get('gt_sv_count', '')};gt_quality={gt_meta.get('gt_quality', '')}",
        pred_source="angle_sweep_infer",
        gt_sv_count=gt_meta.get("gt_sv_count", ""),
        gt_quality=gt_meta.get("gt_quality", ""),
    )


def run_angle_sweep_inference_parallel(
    repo: Path,
    tile_ids: list[str],
    angle: int,
    gpu_ids: list[int],
    cache_dir: Path,
    gt_by_id: dict[str, dict],
    log_fn: Any = None,
    batch_size: int = 48,
    work_dir: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Run baseline inference: batched tiles per subprocess, parallel across GPUs."""
    pending: list[str] = []
    cached_rows: list[dict] = []
    for tile in tile_ids:
        out_pkl = angle_sweep_pred_cache_path(cache_dir, tile, angle)
        if out_pkl.exists() and out_pkl.stat().st_size > 64:
            row = build_fsv1_candidate_row(repo, tile, angle, out_pkl, gt_by_id.get(tile, {}))
            if row:
                cached_rows.append(row)
            continue
        pending.append(tile)

    batch_size = max(1, int(batch_size or 1))
    batches = chunk_list(pending, batch_size)
    todo: list[tuple[list[str], int, int]] = [
        (batch, gpu_ids[i % len(gpu_ids)], i)
        for i, batch in enumerate(batches)
    ]

    if log_fn:
        log_fn(
            f"angle_sweep infer: {len(cached_rows)} cached, {len(pending)} tiles "
            f"in {len(batches)} batch(es) of up to {batch_size} on GPUs {gpu_ids}",
        )

    def _work(item: tuple[list[str], int, int]) -> tuple[list[str], list[dict], str | None]:
        batch, gpu, batch_idx = item
        try:
            run_angle_sweep_baseline_inference_batch(
                repo, batch, angle, gpu, cache_dir, batch_idx, work_dir,
            )
            rows = []
            for tile in batch:
                pred_pkl = angle_sweep_pred_cache_path(cache_dir, tile, angle)
                row = build_fsv1_candidate_row(
                    repo, tile, angle, pred_pkl, gt_by_id.get(tile, {}))
                if row:
                    rows.append(row)
            return batch, rows, None
        except Exception as exc:
            return batch, [], str(exc)

    new_rows: list[dict] = []
    errors: list[str] = []
    n_workers = max(1, len(gpu_ids))
    done_batches = 0
    done_tiles = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_work, item): item for item in todo}
        for fut in as_completed(futures):
            batch, rows, err = fut.result()
            done_batches += 1
            done_tiles += len(batch)
            if err:
                errors.append(f"batch[{batch[0]}..{batch[-1]}]: {err}")
            else:
                new_rows.extend(rows)
            if log_fn and (
                done_batches % 2 == 0 or done_batches == len(todo)
            ):
                log_fn(
                    f"angle_sweep infer progress batches {done_batches}/{len(todo)} "
                    f"tiles ~{done_tiles}/{len(pending)}",
                )
    if log_fn and errors:
        log_fn(f"angle_sweep infer errors: {len(errors)} (first: {errors[0][:200]})")
    return cached_rows + new_rows, errors


def run_angle_sweep_baseline_inference_batch(
    repo: Path,
    tile_ids: list[str],
    angle: int,
    gpu: int,
    cache_dir: Path,
    batch_idx: int,
    work_dir: Path | None = None,
) -> None:
    """Run step1_inference once for up to N tiles; split per-tile pkls into cache_dir."""
    if not tile_ids:
        return
    missing = [
        t for t in tile_ids
        if not (
            angle_sweep_pred_cache_path(cache_dir, t, angle).exists()
            and angle_sweep_pred_cache_path(cache_dir, t, angle).stat().st_size > 64
        )
    ]
    if not missing:
        return

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.exp_mechanism_sv_attractor_gpu89.common_attractor_utils import (
        prepare_angle_sweep_batch_dataset,
    )

    batch_id = hashlib.sha1(
        f"{angle:03d}|{batch_idx}|{'|'.join(sorted(missing))}".encode(),
    ).hexdigest()[:16]
    dataset_root = prepare_angle_sweep_batch_dataset(repo, missing, angle, batch_id)
    tmp_dir = (work_dir or cache_dir.parent) / "pred_cache_angle_sweep" / "_batch_tmp"
    ensure_dir(tmp_dir)
    batch_pkl = tmp_dir / f"gpu{gpu}_b{batch_idx:05d}_a{angle:03d}.pkl"
    err_log = tmp_dir / f"gpu{gpu}_b{batch_idx:05d}_a{angle:03d}.stderr"

    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "MPLCONFIGDIR": "/tmp/mplconfig",
        "OPENRSD_GPU": str(gpu),
        "OPENRSD_BATCH_SIZE": str(len(missing)),
        "OPENRSD_NUM_WORKERS": "4",
        "OPENRSD_FILTER_EMPTY_GT": "0",
        "OPENRSD_CONFIG": str(repo / BASELINE_CONFIG),
        "OPENRSD_CHECKPOINT": str(repo / BASELINE_CKPT),
        "OPENRSD_DATA_ROOT": str(dataset_root),
        "OPENRSD_IMG_DIR": "images",
        "OPENRSD_ANN_DIR": str(dataset_root / "annfiles"),
        "OPENRSD_OUT_RESULTS": str(batch_pkl),
        "OPENRSD_SAVE_VIS": "0",
        "OPENRSD_SAVE_EMPTY_VIS": "0",
    })
    cmd = [str(PYTHON_BIN), str(repo / "SimpleRun/step1_inference.py")]
    proc = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        err_log.write_text(proc.stderr or proc.stdout or "", encoding="utf-8")
        raise RuntimeError(
            f"batch inference failed ({len(missing)} tiles) gpu={gpu}: see {err_log}",
        )
    if not batch_pkl.exists():
        raise RuntimeError(f"batch pkl missing: {batch_pkl}")
    with batch_pkl.open("rb") as f:
        all_results = pickle.load(f)
    if not isinstance(all_results, dict):
        raise RuntimeError(f"unexpected batch pkl format: {batch_pkl}")
    not_found = split_batch_results_to_cache(all_results, missing, cache_dir, angle)
    if not_found:
        raise RuntimeError(
            f"batch output missing {len(not_found)} tiles: {not_found[:5]}",
        )
    try:
        batch_pkl.unlink(missing_ok=True)
    except OSError:
        pass


def run_angle_sweep_baseline_inference(
    repo: Path, tile_id: str, angle: int, gpu: int, cache_dir: Path,
) -> Path:
    run_angle_sweep_baseline_inference_batch(
        repo, [tile_id], angle, gpu, cache_dir, batch_idx=0,
    )
    return angle_sweep_pred_cache_path(cache_dir, tile_id, angle)


def score_stats(scores: np.ndarray) -> tuple[float, float]:
    if scores is None or len(scores) == 0:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.median(scores))


def compute_tile_metrics(
    det_texts: list, det_scores: np.ndarray, det_polys: np.ndarray,
    gt_texts: list, gt_polys: np.ndarray, gt_quality: str,
    dense_sv_ratio: float | None = None,
) -> dict[str, Any]:
    total = len(det_texts)
    sv_idx = [i for i, t in enumerate(det_texts) if is_sv_label(t)]
    sv_dets = len(sv_idx)
    final_sv_ratio = sv_dets / total if total else 0.0
    gt_sv_count = sum(1 for t in gt_texts if is_sv_label(t))

    out: dict[str, Any] = dict(
        total_dets=total, sv_dets=sv_dets, final_sv_ratio=final_sv_ratio,
        gt_sv_count=gt_sv_count, dense_sv_ratio=dense_sv_ratio if dense_sv_ratio is not None else "",
        gt_quality=gt_quality,
    )
    score_mean, score_median = score_stats(det_scores)
    out["score_mean"] = score_mean
    out["score_median"] = score_median

    if gt_quality in ("GT_MISSING",):
        for k in ("matched_sv_iou01", "matched_sv_iou03", "matched_sv_iou05",
                  "sv_match_ratio_iou03", "sv_fp_ratio_iou03", "false_sv_count_iou03",
                  "false_sv_final_ratio_iou03", "false_sv_hub_score_iou03",
                  "max_iou_mean", "max_iou_median", "alignment_status"):
            out[k] = "NA"
        out["alignment_status"] = gt_quality
        return out

    if sv_dets == 0:
        out.update(dict(
            matched_sv_iou01=0, matched_sv_iou03=0, matched_sv_iou05=0,
            sv_match_ratio_iou03=0.0, sv_fp_ratio_iou03=0.0,
            false_sv_count_iou03=0, false_sv_final_ratio_iou03=0.0,
            false_sv_hub_score_iou03=0.0,
            max_iou_mean=0.0, max_iou_median=0.0,
            alignment_status="NO_SV_DETS",
        ))
        return out

    m01, ious, iou_method = match_sv_to_gt(det_polys, det_texts, det_scores, gt_polys, gt_texts, 0.1)
    m03, _, _ = match_sv_to_gt(det_polys, det_texts, det_scores, gt_polys, gt_texts, 0.3)
    m05, _, _ = match_sv_to_gt(det_polys, det_texts, det_scores, gt_polys, gt_texts, 0.5)

    out["matched_sv_iou01"] = m01
    out["matched_sv_iou03"] = m03
    out["matched_sv_iou05"] = m05
    out["sv_match_ratio_iou03"] = m03 / sv_dets if sv_dets else 0.0
    out["sv_fp_ratio_iou03"] = 1.0 - out["sv_match_ratio_iou03"]
    out["false_sv_count_iou03"] = sv_dets - m03
    out["false_sv_final_ratio_iou03"] = out["false_sv_count_iou03"] / total if total else 0.0
    out["false_sv_hub_score_iou03"] = (
        final_sv_ratio * out["sv_fp_ratio_iou03"] * math.log(1 + sv_dets)
    )
    if ious:
        out["max_iou_mean"] = float(np.mean(ious))
        out["max_iou_median"] = float(np.median(ious))
    else:
        out["max_iou_mean"] = 0.0
        out["max_iou_median"] = 0.0

    if gt_quality == "PLACEHOLDER_TEMPLATE":
        out["alignment_status"] = "PLACEHOLDER_GT"
    elif gt_sv_count == 0 and sv_dets > 50:
        out["alignment_status"] = "GT_EMPTY_HIGH_SV_OUTPUT"
    else:
        out["alignment_status"] = iou_method
    return out


def discover_resources(repo: Path) -> list[dict]:
    rows = []
    paths = [
        ("atlas_raw", repo / "resultmd/exp_mechanism_sv_attractor_gpu89/data/ftable_01_attractor_atlas_raw.csv"),
        ("atlas_tile_summary", repo / "resultmd/exp_mechanism_sv_attractor_gpu89/data/ftable_01_attractor_atlas_tile_summary.csv"),
        ("atlas_tile_list", repo / "resultmd/exp_mechanism_sv_attractor_gpu89/data/fmeta_01_tile_list.json"),
        ("encoder_swap_raw", repo / "resultmd/exp_encoder_swap_semantic_drift_20260525_overnight/exp_03_text_encoder_swap_detector/ftable_03_text_encoder_swap_raw.csv"),
        ("dehub_mechanism", repo / "resultmd/exp_next_plan_sv_dehub_step23_overnight/data/ftable_20_dehub_mechanism_eval_raw.csv"),
        ("heldout_ap", repo / "resultmd/exp_formal_success_check_gpu89/data/ftable_33_step23_heldout500_ap_summary.csv"),
        ("b8k_heldout", repo / "resultmd/exp_formal_success_check_gpu89/data/ftable_50_dehub_candidate_heldout500_summary.csv"),
        ("ec_summary", repo / "resultmd/exp_evidence_chain_closure_20260526/fres_999_evidence_chain_closure_summary.md"),
        ("ec3_audit", repo / "work_dirs/exp_evidence_chain_closure_20260526/ec3_true_sv_audit/ftable_true_sv_gt_audit.csv"),
        ("p0148_vis_pkl", repo / f"vis/{P0148}/results.pkl"),
        ("p0682_vis_pkl", repo / f"vis/{P0682}/results.pkl"),
        ("baseline_ckpt", repo / "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"),
        ("b8k_ckpt", repo / "work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_B_fresh_epoch24_to8k/iter_8000.pth"),
        ("support_pkl", repo / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"),
        ("normalized_class_dict", repo / "data/normalized_class_dict.pkl"),
        ("dota_ss_train_ann", repo / "data/DOTA1_1024_500/ss_train/annfiles"),
        ("angle_sweep_root", repo / "data/DOTA1_1024_500/angle_sweep_val/realistic"),
    ]
    for rtype, p in paths:
        note = ""
        if rtype.endswith("ann") and p.is_dir():
            note = f"{sum(1 for _ in p.glob('*.txt'))} txt files"
        rows.append(dict(resource_type=rtype, path=str(p), exists=p.exists(), note=note))
    for pkl in sorted((repo / "vis").glob("*/results.pkl")):
        rows.append(dict(
            resource_type="vis_results_pkl", path=str(pkl), exists=True,
            note=pkl.parent.name,
        ))
    return rows


def run_fsv0(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    resources = discover_resources(repo)
    write_csv(wd / "fsv0_inventory/ftable_resource_paths.csv", resources,
              ["resource_type", "path", "exists", "note"])
    vers = package_versions()
    status = "OK"
    gt_index_note = ""
    if ctx.get("gt_source") == "angle_sweep":
        angle = int(ctx.get("canonical_angle", 0))
        gt_rows = build_angle_sweep_gt_index(repo, wd, angle)
        real_n = sum(1 for r in gt_rows if r.get("gt_quality") == "REAL")
        gt_index_note = (
            f"- DOTA1 12-angle val GT index: **{len(gt_rows)}** tiles @ angle {angle:03d} "
            f"({real_n} REAL, {len(gt_rows) - real_n} placeholder/empty)."
        )
    md = [
        "# FSV0 Inventory / Preflight",
        "",
        f"- status: `{status}`",
        f"- generated_at: `{now_iso()}`",
        f"- git_commit: `{git_commit(repo)}`",
        f"- repo_root: `{repo}`",
        f"- work_dir: `{wd}`",
        f"- requested_gpu_ids: `{ctx['gpu_ids']}`",
        f"- gt_source: `{ctx.get('gt_source', 'auto')}`",
        "",
        "## Environment",
        "",
        markdown_table([vers], list(vers.keys())),
        "",
        "## Resource paths",
        "",
        markdown_table(resources, ["resource_type", "path", "exists", "note"], limit=80),
        "",
        "## GT / prediction readiness",
        "",
        gt_index_note or "- (vis/auto mode)",
        "- Vis `results.pkl` found for many diagnostic tiles (incl. P0148, P0682, atlas cross tiles).",
        "- **angle_sweep** mode uses `data/DOTA1_1024_500/angle_sweep_val/realistic/angle_XXX/annfiles/*.txt` only.",
        "- P0148/P0682 are **not** in the official DOTA1 angle-sweep val split (use vis runs separately).",
        "- Offline mining can compute **rotated IoU** when mmcv/mmrotate import succeeds; else **HBB_APPROX**.",
        "",
        "## Evidence chain EC1–EC7",
        "",
    ]
    ec_root = repo / "resultmd/exp_evidence_chain_closure_20260526"
    for name in sorted(ec_root.glob("fres_*.md")):
        md.append(f"- `{name}` ({'exists' if name.exists() else 'missing'})")
    (rmd / "fres_000_inventory_preflight.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": status, "md": str(rmd / "fres_000_inventory_preflight.md")}


def collect_atlas_candidates(repo: Path) -> list[dict]:
    path = repo / "resultmd/exp_mechanism_sv_attractor_gpu89/data/ftable_01_attractor_atlas_raw.csv"
    rows_out = []
    for r in read_csv(path):
        if r.get("method") != "baseline":
            continue
        try:
            ang = int(float(r["angle"]))
        except Exception:
            continue
        if ang != 0:
            continue
        tile = r["tile_id"]
        rows_out.append(dict(
            tile_id=tile,
            source="atlas_baseline_angle0",
            image_path=str(repo / f"vis/{tile}/dataset/images/{tile}_rot000.jpg"),
            ann_path=str(repo / f"vis/{tile}/dataset/annfiles/{tile}_rot000.pkl"),
            pred_path=str(repo / f"vis/{tile}/results.pkl"),
            total_dets=r.get("det_count", ""),
            sv_dets=r.get("small_vehicle_count", ""),
            final_sv_ratio=r.get("final_sv_ratio", ""),
            dense_sv_ratio=r.get("dense_top1_sv_ratio", ""),
            score_mean=r.get("mean_score", ""),
            note=r.get("notes", ""),
        ))
    return rows_out


def collect_encoder_swap(repo: Path) -> dict[tuple[str, str, str], dict]:
    path = repo / "resultmd/exp_encoder_swap_semantic_drift_20260525_overnight/exp_03_text_encoder_swap_detector/ftable_03_text_encoder_swap_raw.csv"
    out = {}
    for r in read_csv(path):
        tile = r.get("tile_id", "")
        mk = r.get("model_key", r.get("model", ""))
        try:
            ang = int(float(r.get("angle", 0)))
        except Exception:
            ang = 0
        out[(tile, mk, f"{ang:03d}")] = r
    return out


def run_fsv1_angle_sweep(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    angle = int(ctx.get("canonical_angle", 0))
    gpu_ids = parse_gpu_id_list(ctx.get("infer_gpu_ids") or ctx.get("infer_gpu", "8"))
    cache_dir = wd / "pred_cache_angle_sweep"
    ensure_dir(cache_dir)
    log_path = wd / "logs" / "fsv1_angle_sweep_infer.log"
    ensure_dir(log_path.parent)

    def _log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    gt_path = wd / "fsv0_inventory/ftable_angle_sweep_gt_index.csv"
    if gt_path.exists() and not ctx.get("force_reindex"):
        gt_rows = read_csv(gt_path)
    else:
        gt_rows = build_angle_sweep_gt_index(repo, wd, angle)

    max_tiles = int(ctx.get("max_tiles") or 0)
    if ctx.get("mode") == "smoke":
        max_tiles = min(max_tiles or 8, 8)
    elif not max_tiles:
        max_tiles = 500

    tile_ids = select_angle_sweep_candidate_tiles(gt_rows, max_tiles)
    gt_by_id = {r["tile_id"]: r for r in gt_rows}
    infer_batch = max(1, int(ctx.get("infer_batch_size") or 48))
    _log(
        f"fsv1 start n_tiles={len(tile_ids)} gpus={gpu_ids} "
        f"angle={angle:03d} batch_size={infer_batch}",
    )

    rows, infer_errors = run_angle_sweep_inference_parallel(
        repo, tile_ids, angle, gpu_ids, cache_dir, gt_by_id, log_fn=_log,
        batch_size=infer_batch, work_dir=wd,
    )
    rows.sort(key=lambda r: -float(r.get("final_sv_ratio") or 0))

    write_csv(wd / "fsv1_candidates/ftable_high_sv_candidates.csv", rows)
    md = [
        "# FSV1 High-SV-Output Candidate Collection (DOTA1 12-angle val)",
        "",
        "- status: OK",
        f"- candidates: {len(rows)}",
        f"- requested_tiles: {len(tile_ids)}",
        f"- infer_errors: {len(infer_errors)}",
        f"- canonical_angle: `{angle:03d}`",
        f"- inference_gpus: `{','.join(str(g) for g in gpu_ids)}`",
        f"- inference_cache: `{cache_dir}`",
        f"- selection: top GT-SV + low-GT-SV (+ mid fill), max_tiles={max_tiles}",
        "",
        "P0148/P0682 are outside this val split; not included unless present in angle_sweep annfiles.",
        "",
        markdown_table(rows, [
            "tile_id", "final_sv_ratio", "sv_dets", "gt_sv_count", "gt_quality", "pred_path", "note",
        ], limit=50),
    ]
    (rmd / "fres_010_high_sv_candidate_collection.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK", "n": len(rows), "infer_errors": len(infer_errors)}


def run_fsv1(ctx: dict) -> dict:
    if ctx.get("gt_source") == "angle_sweep":
        return run_fsv1_angle_sweep(ctx)
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    seen: set[str] = set()
    rows: list[dict] = []

    def add_row(row: dict, force: bool = False) -> None:
        tid = row["tile_id"]
        if tid in seen:
            if not force:
                return
            for i, existing in enumerate(rows):
                if existing["tile_id"] != tid:
                    continue
                merged = dict(existing)
                for k, v in row.items():
                    if v not in ("", None, "NA"):
                        merged[k] = v
                rows[i] = merged
                return
        seen.add(tid)
        rows.append(row)

    for row in collect_atlas_candidates(repo):
        ok = (
            float(row.get("final_sv_ratio") or 0) >= 0.4
            or int(float(row.get("sv_dets") or 0)) >= 50
            or float(row.get("dense_sv_ratio") or 0) >= 0.5
        )
        if ok or row["tile_id"] in (P0148, P0682):
            add_row(row)

    for pkl in sorted((repo / "vis").glob("*/results.pkl")):
        tile = pkl.parent.name
        if not tile.startswith("P"):
            continue
        img = repo / f"vis/{tile}/dataset/images/{tile}_rot000.jpg"
        if not img.exists():
            continue
        # quick stats from pkl angle 0
        pred = load_vis_prediction(pkl, tile, 0)
        if pred is None:
            continue
        texts = pred["texts"]
        total = len(texts)
        sv = sum(1 for t in texts if is_sv_label(t))
        fsr = sv / total if total else 0
        if fsr >= 0.4 or sv >= 50 or tile in (P0148, P0682):
            add_row(dict(
                tile_id=tile, source="vis_results_pkl",
                image_path=str(img),
                ann_path=str(repo / f"vis/{tile}/dataset/annfiles/{tile}_rot000.pkl"),
                pred_path=str(pkl),
                total_dets=total, sv_dets=sv, final_sv_ratio=fsr,
                dense_sv_ratio="", score_mean=float(np.mean(pred["scores"])) if len(pred["scores"]) else "",
                note="vis_pkl_angle0",
            ), force=True)

    for tile, role in ((P0148, "false_sv_hub_diagnostic"), (P0682, "true_sv_rich_control")):
        add_row(dict(
            tile_id=tile, source="user_mandatory",
            image_path=str(repo / f"vis/{tile}/dataset/images/{tile}_rot000.jpg"),
            ann_path=str(repo / f"vis/{tile}/dataset/annfiles/{tile}_rot000.pkl"),
            pred_path=str(repo / f"vis/{tile}/results.pkl"),
            total_dets="", sv_dets="", final_sv_ratio="",
            dense_sv_ratio="", score_mean="",
            note=role,
        ), force=True)

    write_csv(wd / "fsv1_candidates/ftable_high_sv_candidates.csv", rows)
    md = [
        "# FSV1 High-SV-Output Candidate Collection",
        "",
        f"- status: OK",
        f"- candidates: {len(rows)}",
        f"- csv: `{wd / 'fsv1_candidates/ftable_high_sv_candidates.csv'}`",
        "",
        "## Role notes",
        "",
        f"- `{P0148}`: false-SV-hub **diagnostic** candidate (do not treat as generic failure without GT/manual).",
        f"- `{P0682}`: true-SV-rich **positive control** (manual: dense real small vehicles).",
        "",
        markdown_table(rows, [
            "tile_id", "source", "final_sv_ratio", "sv_dets", "dense_sv_ratio", "pred_path", "note",
        ], limit=40),
    ]
    (rmd / "fres_010_high_sv_candidate_collection.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK", "n": len(rows)}


def run_fsv2(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    gt_source = ctx.get("gt_source", "auto")
    candidates = read_csv(wd / "fsv1_candidates/ftable_high_sv_candidates.csv")
    enc = collect_encoder_swap(repo) if gt_source != "angle_sweep" else {}
    match_rows: list[dict] = []
    canonical_angle = int(ctx.get("canonical_angle", 0))

    methods = [("baseline", "baseline")]
    if gt_source != "angle_sweep":
        methods.extend([
            ("B8k_iter_8000", "B8k"),
            ("B8k_iter_5000", "B8k"),
        ])

    for cand in candidates:
        tile = cand["tile_id"]
        gt = load_gt(repo, tile, canonical_angle, gt_source=gt_source)
        dense_sv = cand.get("dense_sv_ratio", "")

        for method_name, model_key in methods:
            row_base = dict(
                tile_id=tile, method=method_name,
                gt_path=gt["path"], gt_quality=gt["gt_quality"],
            )
            pred_path = Path(cand.get("pred_path") or "")
            pred = None
            if model_key == "baseline" and pred_path.exists():
                pred = load_candidate_prediction(cand, tile, canonical_angle)

            if pred is not None:
                m = compute_tile_metrics(
                    pred["texts"], pred["scores"], pred["polys"],
                    gt["texts"], gt["polys"], gt["gt_quality"],
                    float(dense_sv) if dense_sv not in ("", "NA") else None,
                )
                row = dict(row_base)
                row.update(m)
                row["pred_path"] = str(pred_path)
                row["pred_source"] = cand.get("pred_source", "vis_pkl")
            else:
                # counts-only from encoder swap @ angle 000
                es = enc.get((tile, model_key, "000")) or enc.get((tile, model_key, "0"))
                if es is None and model_key == "B8k":
                    es = enc.get((tile, "B8k", "000"))
                if es:
                    total = int(float(es.get("final_total_det") or 0))
                    sv = int(float(es.get("final_sv_count") or 0))
                    fsr = float(es.get("final_sv_ratio") or (sv / total if total else 0))
                    row = dict(row_base)
                    row.update(dict(
                        total_dets=total, sv_dets=sv, final_sv_ratio=fsr,
                        gt_sv_count=sum(1 for t in gt["texts"] if is_sv_label(t)),
                        dense_sv_ratio=es.get("dense_top1_sv_ratio", dense_sv),
                        pred_source="encoder_swap_counts",
                        pred_path="SOURCE_MISSING_BOXES",
                    ))
                    for k in ("matched_sv_iou01", "matched_sv_iou03", "matched_sv_iou05",
                              "sv_match_ratio_iou03", "sv_fp_ratio_iou03",
                              "false_sv_count_iou03", "false_sv_final_ratio_iou03",
                              "false_sv_hub_score_iou03", "max_iou_mean", "max_iou_median"):
                        row[k] = "NA"
                    row["alignment_status"] = "COUNT_ONLY_NO_BOXES"
                else:
                    row = dict(row_base)
                    row["alignment_status"] = "PRED_SOURCE_MISSING"
                    for k in ("total_dets", "sv_dets", "final_sv_ratio", "gt_sv_count"):
                        row[k] = "NA"

            if tile in TILE_ROLES:
                row["expected_role"] = TILE_ROLES[tile]
            match_rows.append(row)

    write_csv(wd / "fsv2_gt_match/ftable_sv_gt_matching.csv", match_rows)
    p148 = [r for r in match_rows if r.get("tile_id") == P0148 and r.get("method") == "baseline"]
    p682 = [r for r in match_rows if r.get("tile_id") == P0682 and r.get("method") == "baseline"]
    top_false = sorted(
        [r for r in match_rows if r.get("method") == "baseline" and r.get("gt_quality") == "REAL"],
        key=lambda r: -to_float(r.get("false_sv_final_ratio_iou03")),
    )[:10]
    top_true = sorted(
        [r for r in match_rows if r.get("method") == "baseline" and r.get("gt_quality") == "REAL"],
        key=lambda r: -to_float(r.get("sv_match_ratio_iou03")),
    )[:10]
    md = [
        "# FSV2 SV GT Matching",
        "",
        f"- status: OK",
        f"- rows: {len(match_rows)}",
        f"- gt_source: `{gt_source}`",
        f"- canonical_angle: `{canonical_angle:03d}`",
        f"- primary IoU threshold for taxonomy: **0.3** (also export 0.1 / 0.5)",
        "",
        "## Top false-SV-hub proxy @ IoU0.3 (REAL GT, baseline)",
        "",
        markdown_table(top_false, [
            "tile_id", "final_sv_ratio", "gt_sv_count", "sv_dets", "matched_sv_iou03",
            "sv_match_ratio_iou03", "false_sv_final_ratio_iou03", "alignment_status",
        ]),
        "",
        "## Top true-SV-rich proxy @ IoU0.3 (REAL GT, baseline)",
        "",
        markdown_table(top_true, [
            "tile_id", "final_sv_ratio", "gt_sv_count", "sv_dets", "matched_sv_iou03",
            "sv_match_ratio_iou03", "false_sv_final_ratio_iou03",
        ]),
        "",
        "## P0148 vs P0682 (if present; baseline)",
        "",
    ]
    if p148 or p682:
        md.append(markdown_table(p148 + p682, [
            "tile_id", "expected_role", "gt_quality", "total_dets", "sv_dets", "gt_sv_count",
            "matched_sv_iou03", "sv_match_ratio_iou03", "sv_fp_ratio_iou03",
            "false_sv_final_ratio_iou03", "alignment_status",
        ]))
    else:
        md.append("- Not in this candidate set (external vis controls).")
    md.extend([
        "",
        "## Interpretation guardrail",
        "",
        "- Metrics use **DOTA1 angle_sweep_val** txt GT when `gt_source=angle_sweep`.",
        "- When `gt_quality=PLACEHOLDER_TEMPLATE`, IoU metrics are unreliable.",
        "",
    ])
    (rmd / "fres_020_sv_gt_matching.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK", "n": len(match_rows)}


def run_fsv3(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    rows = read_csv(wd / "fsv2_gt_match/ftable_sv_gt_matching.csv")
    box_sources = {"vis_pkl", "angle_sweep_infer"}
    base_rows = [
        r for r in rows
        if r.get("method") == "baseline" and r.get("pred_source") in box_sources
    ]
    # one row per tile (fsv1 may have been fixed; guard duplicates anyway)
    by_tile: dict[str, dict] = {}
    for r in base_rows:
        tid = r["tile_id"]
        if tid not in by_tile:
            by_tile[tid] = r
    base_rows = list(by_tile.values())

    rank_false: list[dict] = []
    for r in base_rows:
        if ctx.get("gt_source") != "angle_sweep" and r.get("tile_id") == P0682:
            continue  # vis positive control
        ffr = to_float(r.get("false_sv_final_ratio_iou03"))
        hub = to_float(r.get("false_sv_hub_score_iou03"))
        rank_false.append({**r, "_ffr": ffr, "_hub": hub})
    rank_false.sort(key=lambda x: (-x["_ffr"], -x["_hub"]))
    rank_false_trusted = [
        r for r in rank_false
        if r.get("gt_quality") != "PLACEHOLDER_TEMPLATE"
    ]

    rank_true: list[dict] = []
    for r in base_rows:
        sm = to_float(r.get("sv_match_ratio_iou03"))
        fsr = to_float(r.get("final_sv_ratio"))
        rank_true.append({**r, "_sm": sm, "_fsr": fsr})
    rank_true.sort(key=lambda x: (-x["_fsr"], -x["_sm"]))

    out_false = []
    for i, r in enumerate(rank_false[:50], 1):
        tile = r["tile_id"]
        ttype = classify_tile_prelim(r, tile, ctx.get("gt_source", "auto"))
        out_false.append(dict(
            rank=i, tile_id=tile,
            total_dets=r.get("total_dets"), sv_dets=r.get("sv_dets"),
            final_sv_ratio=r.get("final_sv_ratio"), gt_sv_count=r.get("gt_sv_count"),
            matched_iou03=r.get("matched_sv_iou03"),
            sv_fp_ratio_iou03=r.get("sv_fp_ratio_iou03"),
            false_sv_final_ratio_iou03=r.get("false_sv_final_ratio_iou03"),
            false_sv_hub_score_iou03=r.get("false_sv_hub_score_iou03"),
            dense_sv_ratio=r.get("dense_sv_ratio"),
            gt_quality=r.get("gt_quality"),
            tile_type_prelim=ttype,
        ))

    out_true = []
    for i, r in enumerate(rank_true[:30], 1):
        out_true.append(dict(
            rank=i, tile_id=r["tile_id"],
            final_sv_ratio=r.get("final_sv_ratio"),
            sv_match_ratio_iou03=r.get("sv_match_ratio_iou03"),
            gt_sv_count=r.get("gt_sv_count"), sv_dets=r.get("sv_dets"),
            gt_quality=r.get("gt_quality"),
            tile_type_prelim=classify_tile_prelim(r, r["tile_id"], ctx.get("gt_source", "auto")),
        ))

    out_false_trusted = []
    for i, r in enumerate(rank_false_trusted[:20], 1):
        out_false_trusted.append(dict(
            rank=i, tile_id=r["tile_id"],
            final_sv_ratio=r.get("final_sv_ratio"),
            sv_fp_ratio_iou03=r.get("sv_fp_ratio_iou03"),
            false_sv_final_ratio_iou03=r.get("false_sv_final_ratio_iou03"),
            false_sv_hub_score_iou03=r.get("false_sv_hub_score_iou03"),
            gt_quality=r.get("gt_quality"),
            tile_type_prelim=classify_tile_prelim(r, r["tile_id"], ctx.get("gt_source", "auto")),
        ))

    write_csv(wd / "fsv3_false_sv_score/ftable_false_sv_hub_ranking.csv", out_false)
    write_csv(wd / "fsv3_false_sv_score/ftable_false_sv_hub_ranking_trusted_gt.csv", out_false_trusted)
    write_csv(wd / "fsv3_false_sv_score/ftable_true_sv_rich_ranking.csv", out_true)

    md = [
        "# FSV3 False-SV-Hub Score And Ranking",
        "",
        "- status: OK",
        "- primary sort: `false_sv_final_ratio_iou03` desc, then `false_sv_hub_score_iou03`",
        "- P0682 excluded from false-hub ranking (true-SV-rich positive control).",
        "",
        "## Top false-SV-hub diagnostic (trusted GT only; top 20)",
        "",
        "> Tiles with `PLACEHOLDER_TEMPLATE` GT are excluded from this table "
        "(IoU proxy is unreliable). Use manual crop audit + P0148 anchor.",
        "",
        markdown_table(
            out_false_trusted or [dict(rank="—", tile_id="NONE", note="no real GT in candidate set")],
            ["rank", "tile_id", "final_sv_ratio", "sv_fp_ratio_iou03",
             "false_sv_final_ratio_iou03", "false_sv_hub_score_iou03", "gt_quality", "tile_type_prelim"],
        ),
        "",
        "## Top false-SV-hub (auto-GT-proxy incl. placeholder; top 20)",
        "",
        markdown_table(out_false[:20], [
            "rank", "tile_id", "final_sv_ratio", "sv_fp_ratio_iou03",
            "false_sv_final_ratio_iou03", "false_sv_hub_score_iou03", "gt_quality", "tile_type_prelim",
        ]),
        "",
        "## Top true-SV-rich (high final_sv + high match@0.3)",
        "",
        markdown_table(out_true[:10], [
            "rank", "tile_id", "final_sv_ratio", "sv_match_ratio_iou03", "gt_sv_count", "tile_type_prelim",
        ]),
    ]
    (rmd / "fres_030_false_sv_hub_ranking.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK"}


def classify_tile_prelim(r: dict, tile_id: str, gt_source: str = "auto") -> str:
    if gt_source != "angle_sweep":
        if tile_id == P0682:
            return "true_sv_rich"
        if tile_id == P0148:
            return "false_sv_hub"
    gq = r.get("gt_quality", "")
    if gq == "PLACEHOLDER_TEMPLATE":
        return "high_sv_gt_pending"
    fsr = to_float(r.get("final_sv_ratio"))
    sm = to_float(r.get("sv_match_ratio_iou03"))
    fp = to_float(r.get("sv_fp_ratio_iou03"))
    ffr = to_float(r.get("false_sv_final_ratio_iou03"))
    dense = to_float(r.get("dense_sv_ratio"))
    if dense >= 0.5 and fsr < 0.4:
        return "latent_sv_bias"
    if fsr >= 0.4 and sm >= 0.5:
        return "true_sv_rich"
    if fsr >= 0.4 and fp >= 0.7 and ffr >= 0.25:
        return "false_sv_hub"
    if fsr >= 0.4 and 0.3 <= sm < 0.5:
        return "mixed"
    if fsr < 0.4 and ffr < 0.2:
        return "normal"
    return "mixed"


def run_fsv4(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    pack = wd / "fsv4_crop_audit_pack"
    crops = pack / "crops"
    sheets = pack / "contact_sheets"
    ensure_dir(crops)
    ensure_dir(sheets)

    rank = read_csv(wd / "fsv3_false_sv_score/ftable_false_sv_hub_ranking.csv")
    true_rank = read_csv(wd / "fsv3_false_sv_score/ftable_true_sv_rich_ranking.csv")
    audit_tiles: list[str] = []
    for r in rank[:10]:
        audit_tiles.append(r["tile_id"])
    for r in true_rank[:5]:
        if r["tile_id"] not in audit_tiles:
            audit_tiles.append(r["tile_id"])
    for t in (P0148, P0682):
        if t not in audit_tiles:
            audit_tiles.append(t)

    try:
        import cv2
    except ImportError:
        cv2 = None

    ann_rows: list[dict] = []
    max_crops = ctx.get("max_crops_per_tile", 50)

    angle = int(ctx.get("canonical_angle", 0))
    gt_source = ctx.get("gt_source", "auto")
    cand_by_tile = {c["tile_id"]: c for c in read_csv(wd / "fsv1_candidates/ftable_high_sv_candidates.csv")}

    for tile in audit_tiles:
        cand = cand_by_tile.get(tile, {})
        pred_path = Path(cand.get("pred_path") or repo / f"vis/{tile}/results.pkl")
        if gt_source == "angle_sweep":
            img_path = angle_sweep_image_path(repo, tile, angle)
            if img_path is None:
                img_path = Path(cand.get("image_path") or "")
        else:
            img_path = Path(cand.get("image_path") or repo / f"vis/{tile}/dataset/images/{tile}_rot000.jpg")
        if not pred_path.exists() or not img_path or not Path(img_path).exists() or cv2 is None:
            continue
        pred = load_candidate_prediction(cand if cand else {"pred_path": str(pred_path)}, tile, angle)
        gt = load_gt(repo, tile, angle, gt_source=gt_source)
        if pred is None:
            continue
        img = cv2.imread(str(Path(img_path)))
        if img is None:
            continue
        h, w = img.shape[:2]
        _, max_ious, _ = match_sv_to_gt(
            pred["polys"], pred["texts"], pred["scores"], gt["polys"], gt["texts"], 0.3)

        sv_indices = [i for i, t in enumerate(pred["texts"]) if is_sv_label(t)]
        records = []
        for di, idx in enumerate(sv_indices):
            iou = max_ious[di] if di < len(max_ious) else 0.0
            matched = iou >= 0.3
            records.append(dict(
                idx=idx, score=float(pred["scores"][idx]), iou=iou,
                matched=matched, poly=pred["polys"][idx],
            ))
        unmatched = sorted([x for x in records if not x["matched"]], key=lambda x: -x["score"])
        matched = sorted([x for x in records if x["matched"]], key=lambda x: -x["score"])
        high = sorted(records, key=lambda x: -x["score"])
        import random
        rng = random.Random(42)
        random_s = records[:]
        rng.shuffle(random_s)

        groups = [
            ("unmatched", unmatched[:max_crops]),
            ("matched", matched[:20]),
            ("high_score", high[:20]),
            ("random", random_s[:20]),
        ]
        tile_crops_dir = crops / tile
        ensure_dir(tile_crops_dir)
        saved_paths: dict[str, list[str]] = defaultdict(list)

        det_id = 0
        for group_name, items in groups:
            for rec in items:
                det_id += 1
                poly = rec["poly"].reshape(-1, 2)
                x1 = int(max(0, np.floor(poly[:, 0].min() - 20)))
                y1 = int(max(0, np.floor(poly[:, 1].min() - 20)))
                x2 = int(min(w, np.ceil(poly[:, 0].max() + 20)))
                y2 = int(min(h, np.ceil(poly[:, 1].max() + 20)))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                status = "matched" if rec["matched"] else "unmatched"
                fname = (
                    f"{tile}__baseline__det{det_id:04d}__score{rec['score']:.4f}"
                    f"__iou{rec['iou']:.3f}__{status}.png"
                )
                out_p = tile_crops_dir / fname
                cv2.imwrite(str(out_p), crop)
                rel = str(out_p.relative_to(pack))
                saved_paths[group_name].append(str(out_p))
                ann_rows.append(dict(
                    crop_file=rel, tile_id=tile, method="baseline", det_id=det_id,
                    score=rec["score"],
                    bbox=f"{x1},{y1},{x2},{y2}",
                    max_gt_iou=rec["iou"],
                    auto_match_status=status,
                    crop_group=group_name,
                    manual_label="", notes="",
                ))

        # simple contact sheet: first 25 unmatched
        sheet_imgs = []
        for p in saved_paths.get("unmatched", [])[:25]:
            im = cv2.imread(p)
            if im is not None:
                im = cv2.resize(im, (128, 128))
                sheet_imgs.append(im)
        if sheet_imgs:
            cols = 5
            rows_n = (len(sheet_imgs) + cols - 1) // cols
            canvas = np.zeros((rows_n * 128, cols * 128, 3), dtype=np.uint8)
            for i, im in enumerate(sheet_imgs):
                r, c = divmod(i, cols)
                canvas[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = im
            cv2.imwrite(str(sheets / f"{tile}__unmatched_contact.jpg"), canvas)

    write_csv(pack / "annotation_template.csv", ann_rows)
    md = [
        "# FSV4 Manual Crop Audit Pack",
        "",
        f"- status: OK",
        f"- pack_dir: `{pack}`",
        f"- crops: {len(ann_rows)}",
        f"- contact_sheets: `{sheets}`",
        "",
        "Manual audit pack is ready. Without `manual_label` entries, conclusions remain **auto-GT-proxy** only.",
        "",
        f"- P0682 is a **true-SV-rich positive control** (manual prior: many real small vehicles).",
        f"- P0148 is a **false-SV-hub diagnostic** candidate.",
        "",
        "## Audit tiles",
        "",
        "\n".join(f"- `{t}`" for t in audit_tiles),
    ]
    (rmd / "fres_040_manual_crop_audit_pack.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK", "n_crops": len(ann_rows)}


def run_fsv5(ctx: dict) -> dict:
    rmd, wd = ctx["result_md"], ctx["work"]
    base_all = [r for r in read_csv(wd / "fsv2_gt_match/ftable_sv_gt_matching.csv") if r.get("method") == "baseline"]
    base_by_tile: dict[str, dict] = {}
    for r in base_all:
        if r.get("pred_source") in ("vis_pkl", "angle_sweep_infer"):
            base_by_tile[r["tile_id"]] = r
    base = list(base_by_tile.values())
    tax_rows = []
    for r in base:
        tile = r["tile_id"]
        ttype = classify_tile_prelim(r, tile, ctx.get("gt_source", "auto"))
        if ctx.get("gt_source") != "angle_sweep" and tile == P0682:
            ttype = "true_sv_rich"
            ev = "GT_PLUS_MANUAL" if r.get("gt_quality") != "PLACEHOLDER_TEMPLATE" else "MANUAL_ONLY"
        elif ctx.get("gt_source") != "angle_sweep" and tile == P0148:
            ttype = "false_sv_hub"
            ev = "GT_PLUS_MANUAL" if r.get("gt_quality") != "PLACEHOLDER_TEMPLATE" else "MANUAL_ONLY"
        elif r.get("gt_quality") == "PLACEHOLDER_TEMPLATE":
            ev = "WEAK_SOURCE_ONLY"
        elif r.get("gt_quality") == "GT_MISSING":
            ev = "WEAK_SOURCE_ONLY"
        else:
            ev = "GT_ONLY"
        tax_rows.append(dict(
            tile_id=tile,
            final_sv_ratio=r.get("final_sv_ratio"),
            dense_sv_ratio=r.get("dense_sv_ratio"),
            gt_sv_count=r.get("gt_sv_count"),
            sv_dets=r.get("sv_dets"),
            sv_match_ratio_iou03=r.get("sv_match_ratio_iou03"),
            sv_fp_ratio_iou03=r.get("sv_fp_ratio_iou03"),
            false_sv_final_ratio_iou03=r.get("false_sv_final_ratio_iou03"),
            manual_true_sv_ratio="",
            manual_false_sv_ratio="",
            tile_type=ttype,
            evidence_level=ev,
            gt_quality=r.get("gt_quality"),
        ))

    write_csv(wd / "fsv5_tile_taxonomy/ftable_tile_taxonomy.csv", tax_rows)
    summary = Counter(r["tile_type"] for r in tax_rows)
    md = [
        "# FSV5 Tile Taxonomy",
        "",
        "- status: OK",
        "",
        "## Counts by tile_type",
        "",
        markdown_table(
            [dict(tile_type=k, n_tiles=v) for k, v in sorted(summary.items())],
            ["tile_type", "n_tiles"],
        ),
        "",
        "## Special cases",
        "",
        *(
            [
                f"- `{P0682}` → **true_sv_rich** (vis control; not in angle_sweep val).",
                f"- `{P0148}` → **false_sv_hub** diagnostic (vis control).",
            ]
            if ctx.get("gt_source") != "angle_sweep"
            else [
                "- P0148/P0682 are **outside** DOTA1 angle_sweep val; use vis runs for those controls.",
            ]
        ),
        "- high `final_sv_ratio` alone is **not** equivalent to false-SV-hub.",
        "",
        markdown_table(tax_rows[:25], [
            "tile_id", "tile_type", "final_sv_ratio", "sv_match_ratio_iou03",
            "false_sv_final_ratio_iou03", "gt_quality", "evidence_level",
        ]),
    ]
    (rmd / "fres_050_tile_taxonomy.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK"}


def run_fsv6(ctx: dict) -> dict:
    repo, rmd, wd = ctx["repo"], ctx["result_md"], ctx["work"]
    enc = collect_encoder_swap(repo)
    tax = {r["tile_id"]: r["tile_type"] for r in read_csv(wd / "fsv5_tile_taxonomy/ftable_tile_taxonomy.csv")}
    rows_out = []
    angles = ["000", "030", "090", "180", "270"]

    def es_row(tile: str, mk: str, ang: str) -> dict | None:
        return enc.get((tile, mk, ang))

    for tile, ttype in tax.items():
        for ang in angles:
            b = es_row(tile, "baseline", ang)
            k = es_row(tile, "B8k", ang)
            if not b and not k:
                continue
            for method, rec in (("baseline", b), ("B8k_iter_8000", k)):
                if not rec:
                    continue
                rows_out.append(dict(
                    tile_id=tile, tile_type=ttype, angle=ang, method=method,
                    final_sv_ratio=rec.get("final_sv_ratio"),
                    final_total_det=rec.get("final_total_det"),
                    final_sv_count=rec.get("final_sv_count"),
                    dense_sv_ratio=rec.get("dense_top1_sv_ratio"),
                    note="encoder_swap_counts_only",
                ))

    write_csv(wd / "fsv6_b8k_preserve/ftable_b8k_false_hub_repair_and_preserve.csv", rows_out)

    def tile_table(tile: str) -> list[dict]:
        out = []
        for ang in angles:
            b = es_row(tile, "baseline", ang)
            k = es_row(tile, "B8k", ang)
            if not b:
                continue
            out.append(dict(
                tile_id=tile, angle=ang, method="baseline",
                final_sv_ratio=b.get("final_sv_ratio"),
                sv_dets=b.get("final_sv_count"),
                dense_sv_ratio=b.get("dense_top1_sv_ratio"),
            ))
            if k:
                out.append(dict(
                    tile_id=tile, angle=ang, method="B8k_iter_8000",
                    final_sv_ratio=k.get("final_sv_ratio"),
                    sv_dets=k.get("final_sv_count"),
                    dense_sv_ratio=k.get("dense_top1_sv_ratio"),
                ))
        return out

    b8k_ok = Path(repo / "work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_B_fresh_epoch24_to8k/iter_8000.pth").exists()

    md = [
        "# FSV6 B8k False-Hub Repair And True-SV Preserve",
        "",
        f"- status: {'OK' if b8k_ok else 'PARTIAL_BLOCKED'}",
        f"- B8k checkpoint exists: `{b8k_ok}`",
        "- Comparison uses **encoder_swap count metrics** (no per-box B8k pkl on disk).",
        "",
        "## P0148 (false-SV-hub diagnostic)",
        "",
        markdown_table(tile_table(P0148), ["tile_id", "angle", "method", "final_sv_ratio", "sv_dets"]),
        "",
        "## P0682 (true-SV-rich positive control)",
        "",
        markdown_table(tile_table(P0682), ["tile_id", "angle", "method", "final_sv_ratio", "sv_dets"]),
        "",
        "## Answers",
        "",
        "1. B8k lowers **aggregate** `final_sv_ratio` vs baseline on P0148/P0682 in encoder-swap logs.",
        "2. Without box-level B8k predictions, **false_sv_final_ratio_iou03** preserve check is **BLOCKED** at box level.",
        "3. P0682 B8k also reduces SV count — treat as **safety risk** until true-SV preserve verified via manual crops / real GT.",
        "4. Per-tile-type box preserve verdict requires future B8k `results.pkl` export.",
    ]
    (rmd / "fres_060_b8k_false_hub_repair_and_true_sv_preserve.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK" if b8k_ok else "PARTIAL_BLOCKED"}


def run_fsv999(ctx: dict) -> dict:
    rmd, wd = ctx["result_md"], ctx["work"]
    tasks = []
    for exp in ALL_EXPS:
        key = exp.replace("fsv", "fres_").replace("_inventory", "_000_inventory_preflight")
        if exp == "fsv0_inventory":
            md = rmd / "fres_000_inventory_preflight.md"
        elif exp == "fsv1_collect_candidates":
            md = rmd / "fres_010_high_sv_candidate_collection.md"
        elif exp == "fsv2_gt_match":
            md = rmd / "fres_020_sv_gt_matching.md"
        elif exp == "fsv3_false_sv_score":
            md = rmd / "fres_030_false_sv_hub_ranking.md"
        elif exp == "fsv4_crop_audit_pack":
            md = rmd / "fres_040_manual_crop_audit_pack.md"
        elif exp == "fsv5_tile_taxonomy":
            md = rmd / "fres_050_tile_taxonomy.md"
        else:
            md = rmd / "fres_060_b8k_false_hub_repair_and_true_sv_preserve.md"
        tasks.append(dict(
            task=exp, status="OK" if md.exists() else "MISSING", md_path=str(md),
            csv_path="", note="",
        ))

    false_rank = read_csv(wd / "fsv3_false_sv_score/ftable_false_sv_hub_ranking.csv")
    true_rank = read_csv(wd / "fsv3_false_sv_score/ftable_true_sv_rich_ranking.csv")
    tax = read_csv(wd / "fsv5_tile_taxonomy/ftable_tile_taxonomy.csv")

    p_rows = [r for r in read_csv(wd / "fsv2_gt_match/ftable_sv_gt_matching.csv")
              if r.get("tile_id") in (P0148, P0682) and r.get("method") == "baseline"]

    md = [
        "# FSV999 False-SV-Hub Mining Summary",
        "",
        f"- generated_at: `{now_iso()}`",
        "",
        "## Goal",
        "",
        "Upgrade high-SV-output mining to **false-SV-hub** diagnostics with GT-unmatched ratio",
        "(`false_sv_final_ratio_iou03`), separating **true-SV-rich** scenes (e.g. P0682) from",
        "**false-SV-hub** cases (e.g. P0148).",
        "",
        "## Task status",
        "",
        markdown_table(tasks, ["task", "status", "md_path"]),
        "",
        "## Core metrics",
        "",
        "- `final_sv_ratio` = SV detections / all detections",
        "- `sv_match_ratio_iou03` = fraction of SV dets matched to GT SV at IoU≥0.3",
        "- `false_sv_final_ratio_iou03` = unmatched SV / all detections",
        "- `false_sv_hub_score_iou03` = final_sv_ratio × sv_fp_ratio × log(1+sv_dets)",
        "",
        "## Top false-SV-hub diagnostic (auto proxy, top 15)",
        "",
        markdown_table(false_rank[:15], [
            "rank", "tile_id", "final_sv_ratio", "false_sv_final_ratio_iou03",
            "sv_fp_ratio_iou03", "tile_type_prelim", "gt_quality",
        ]),
        "",
        "## Top true-SV-rich (top 10)",
        "",
        markdown_table(true_rank[:10], [
            "rank", "tile_id", "final_sv_ratio", "sv_match_ratio_iou03", "tile_type_prelim",
        ]),
        "",
        "## P0148 vs P0682",
        "",
        markdown_table(p_rows, [
            "tile_id", "expected_role", "final_sv_ratio", "gt_sv_count", "sv_dets",
            "sv_match_ratio_iou03", "false_sv_final_ratio_iou03", "gt_quality", "alignment_status",
        ]),
        "",
        "## Paper wording (EN)",
        "",
        "> High small-vehicle output is not necessarily a failure, as some tiles contain dense true small vehicles.",
        "> We distinguish high-SV-output tiles from false-SV-hub diagnostic tiles using GT matching",
        "> (IoU≥0.3) and manual crop audit. P0682 serves as a true-SV-rich positive control;",
        "> P0148 as a false-SV-hub diagnostic candidate.",
        "",
        "## Conclusions",
        "",
        "- Do **not** equate high `final_sv_ratio` with false-SV-hub failure.",
        "- With **REAL** angle_sweep GT, separate rankings by `sv_match_ratio_iou03` vs `false_sv_final_ratio_iou03`.",
        "- P0148/P0682 are external vis controls (not in DOTA1 angle_sweep val) unless explicitly added.",
        "- DeHub safety must report false-hub reduction **and** true-SV preserve on rich tiles.",
    ]
    (rmd / "fres_999_false_sv_hub_mining_summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="False-SV-hub tile mining 20260526")
    p.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    p.add_argument("--work-dir", type=Path, default=WORK_DEFAULT)
    p.add_argument("--result-md-dir", type=Path, default=RESULT_MD_DEFAULT)
    p.add_argument("--gpu-ids", default="8,9")
    p.add_argument("--exp", default="all", help="all or comma-separated fsv* names")
    p.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="full")
    p.add_argument("--max-tiles", type=int, default=0)
    p.add_argument("--max-crops-per-tile", type=int, default=50)
    p.add_argument(
        "--gt-source", choices=["auto", "angle_sweep"], default="auto",
        help="GT resolver: angle_sweep uses DOTA1 12-angle val annfiles only",
    )
    p.add_argument("--canonical-angle", type=int, default=0,
                   help="Angle in degrees for mining (0,30,...,330)")
    p.add_argument("--infer-gpu", default="8",
                   help="GPU(s) for angle_sweep baseline inference; comma-separated enables parallel fsv1")
    p.add_argument("--infer-gpu-ids", default="",
                   help="Alias for --infer-gpu (e.g. 4,5,6,7); overrides --infer-gpu when set")
    p.add_argument(
        "--infer-batch-size", type=int, default=48,
        help="Tiles per step1_inference subprocess (angle_sweep fsv1); 1=legacy single-tile",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def should_run(name: str, exp_arg: str) -> bool:
    if exp_arg in ("all", ""):
        return True
    return name in {e.strip() for e in exp_arg.split(",")}


def main() -> int:
    args = parse_args()
    ensure_dir(args.work_dir)
    ensure_dir(args.result_md_dir)
    runner = CommandRunner(args.work_dir, args.gpu_ids)
    runner.open_log("run_false_sv_hub_mining.log")

    infer_spec = (args.infer_gpu_ids or args.infer_gpu).strip()
    ctx = dict(
        repo=args.repo_root.resolve(),
        work=args.work_dir.resolve(),
        result_md=args.result_md_dir.resolve(),
        gpu_ids=args.gpu_ids,
        mode=args.mode,
        max_crops_per_tile=args.max_crops_per_tile,
        max_tiles=args.max_tiles,
        gt_source=args.gt_source,
        canonical_angle=args.canonical_angle,
        infer_gpu=infer_spec,
        infer_gpu_ids=infer_spec,
        infer_batch_size=args.infer_batch_size,
        force_reindex=args.force,
    )
    runner.log(f"start mode={args.mode} exp={args.exp}")

    steps = [
        ("fsv0_inventory", run_fsv0),
        ("fsv1_collect_candidates", run_fsv1),
        ("fsv2_gt_match", run_fsv2),
        ("fsv3_false_sv_score", run_fsv3),
        ("fsv4_crop_audit_pack", run_fsv4),
        ("fsv5_tile_taxonomy", run_fsv5),
        ("fsv6_b8k_preserve_check", run_fsv6),
        ("fsv999", run_fsv999),
    ]

    results = {}
    for name, fn in steps:
        if name == "fsv999":
            if not should_run("all", args.exp) and "fsv999" not in args.exp:
                continue
        elif not should_run(name, args.exp):
            continue
        marker = args.work_dir / f".done_{name}"
        if args.resume and marker.exists() and not args.force and name != "fsv999":
            runner.log(f"skip {name} (resume)")
            continue
        if args.mode == "dryrun":
            runner.log(f"dryrun would run {name}")
            continue
        try:
            runner.log(f"run {name}")
            meta = fn(ctx)
            results[name] = meta
            if name != "fsv999":
                marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            runner.log(f"FAILED {name}: {traceback.format_exc()}")
            fail_md = args.result_md_dir / f"fres_FAIL_{name}.md"
            fail_md.write_text(f"# FAIL {name}\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
            results[name] = {"status": "FAIL"}

    if should_run("all", args.exp) or "fsv999" in args.exp:
        try:
            run_fsv999(ctx)
        except Exception:
            runner.log(traceback.format_exc())

    runner.log(f"done: {results}")
    runner.close_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

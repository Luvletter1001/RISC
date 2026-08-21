#!/usr/bin/env python3
"""Rotation GT-shift taxonomy (Case-1 / Case-2) on angle_sweep_val 12×30°.

Protocol A: co-rotated GT per angle. Inference batch=48, multi-GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_DEFAULT = Path("/data1/zcy/OpenRSD")
WORK_DEFAULT = REPO_DEFAULT / "work_dirs/exp_rotation_gt_shift_taxonomy_20260527"
RESULT_MD_DEFAULT = REPO_DEFAULT / "resultmd/exp_rotation_gt_shift_taxonomy_20260527"
MINING_WD = REPO_DEFAULT / "work_dirs/exp_false_sv_hub_mining_20260526_dota12angle"

ANGLES_12 = list(range(0, 360, 30))
SCORE_THR = 0.3
IOU_CASE1 = 0.3
IOU_WEAK = 0.1
IOU_BG = 0.1
SHIFT_PX = 12.0
DIoU_DROP = 0.15
MIN_GT_AREA = 64.0
# GT classes excluded from Case-1 (SV pred on these non-SV GT is treated as sv_bg).
CASE1_EXCLUDE_GT_CLASSES = frozenset({"large-vehicle"})

# Import shared utilities from mining pipeline
sys.path.insert(0, str(REPO_DEFAULT))
from M_Tools.analysis.run_false_sv_hub_mining_20260526 import (  # noqa: E402
    ANGLE_SWEEP_REL,
    BASELINE_CKPT,
    BASELINE_CONFIG,
    PYTHON_BIN,
    angle_sweep_ann_path,
    angle_sweep_pred_cache_path,
    chunk_list,
    ensure_dir,
    is_sv_label,
    load_angle_sweep_prediction,
    now_iso,
    parse_dota_txt,
    parse_gpu_id_list,
    polys_to_rboxes,
    poly8_to_rbox,
    read_csv,
    run_angle_sweep_inference_parallel,
    to_float,
    write_csv,
)


def is_non_sv_class(cls: str) -> bool:
    return not is_sv_label(cls)


def is_case1_eligible_gt(cls: str) -> bool:
    """Non-SV GT that may enter Case-1 when hit by an SV pred (IoU>=IOU_CASE1)."""
    return is_non_sv_class(cls) and cls not in CASE1_EXCLUDE_GT_CLASSES


def poly_centroid(poly: np.ndarray) -> tuple[float, float]:
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def poly_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def iou_matrix_rotated(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix [len(a), len(b)] between poly sets."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ar = polys_to_rboxes(a)
    br = polys_to_rboxes(b)
    out = np.zeros((len(ar), len(br)), dtype=np.float32)
    try:
        import torch
        from mmcv.ops import box_iou_rotated
        bt = torch.from_numpy(br.astype(np.float32))
        for i, d in enumerate(ar):
            dt = torch.from_numpy(d.reshape(1, 5).astype(np.float32))
            ious = box_iou_rotated(dt, bt).cpu().numpy().reshape(-1)
            out[i] = ious
    except Exception:
        for i, d in enumerate(ar):
            dx1, dy1 = d[0] - d[2] / 2, d[1] - d[3] / 2
            dx2, dy2 = d[0] + d[2] / 2, d[1] + d[3] / 2
            for j, g in enumerate(br):
                gx1, gy1 = g[0] - g[2] / 2, g[1] - g[3] / 2
                gx2, gy2 = g[0] + g[2] / 2, g[1] + g[3] / 2
                ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (dx2 - dx1) * (dy2 - dy1) + (gx2 - gx1) * (gy2 - gy1) - inter + 1e-6
                out[i, j] = inter / union
    return out


def load_pred(repo: Path, cache_dir: Path, tile: str, angle: int) -> dict[str, Any]:
    pkl = angle_sweep_pred_cache_path(cache_dir, tile, angle)
    rec = load_angle_sweep_prediction(pkl, tile)
    if rec is None:
        return dict(texts=[], scores=np.zeros(0), polys=np.zeros((0, 8)))
    scores = rec["scores"]
    keep = scores >= SCORE_THR if len(scores) else np.array([], dtype=bool)
    texts = [t for t, k in zip(rec["texts"], keep) if k]
    polys = rec["polys"][keep] if len(rec["polys"]) else np.zeros((0, 8))
    sc = scores[keep] if len(scores) else np.zeros(0)
    return dict(texts=texts, scores=sc, polys=polys)


def load_gt(tile: str, angle: int, repo: Path) -> dict[str, Any]:
    path = angle_sweep_ann_path(repo, tile, angle)
    texts, polys = parse_dota_txt(path)
    return dict(texts=texts, polys=polys, path=str(path))


def gt_track_map(gt0: dict, gt_a: dict) -> list[int]:
    """Map gt index at angle -> index at angle 0 (-1 if unmatched)."""
    if len(gt0["polys"]) == 0:
        return [-1] * len(gt_a["polys"])
    iou = iou_matrix_rotated(gt_a["polys"], gt0["polys"])
    mapping = []
    for j in range(len(gt_a["polys"])):
        if iou.shape[0] <= j:
            mapping.append(-1)
            continue
        best_i = int(iou[j].argmax()) if iou.shape[1] else -1
        mapping.append(best_i if best_i >= 0 and iou[j, best_i] >= IOU_WEAK else -1)
    return mapping


def best_pred_for_gt(
    gt_poly: np.ndarray, pred: dict,
) -> tuple[int, float, str]:
    if len(pred["polys"]) == 0:
        return -1, 0.0, ""
    iou = iou_matrix_rotated(gt_poly.reshape(1, 8), pred["polys"])[0]
    j = int(iou.argmax())
    return j, float(iou[j]), pred["texts"][j]


def run_r0_cohort(ctx: dict) -> dict:
    repo, wd = ctx["repo"], ctx["work"]
    n = int(ctx["cohort_size"])
    seed = int(ctx.get("seed", 20260527))
    rng = random.Random(seed)

    ann_dir = repo / ANGLE_SWEEP_REL / "angle_000" / "annfiles"
    all_tiles = sorted(p.stem for p in ann_dir.glob("*.txt"))

    mining_path = MINING_WD / "fsv1_candidates/ftable_high_sv_candidates.csv"
    mining_tiles: list[str] = []
    if mining_path.exists():
        mining_tiles = [r["tile_id"] for r in read_csv(mining_path)]

    cohort_path = wd / f"ftable_cohort_{n}.csv"
    existing: list[str] = []
    if ctx.get("extend_cohort"):
        for prev_n in (500, 1000, 1500, n):
            prev_p = wd / f"ftable_cohort_{prev_n}.csv"
            if prev_p.exists() and prev_n < n:
                existing = [r["tile_id"] for r in read_csv(prev_p)]
                break
        if not existing and cohort_path.exists():
            existing = [r["tile_id"] for r in read_csv(cohort_path)]

    chosen: list[str] = []
    if existing:
        chosen = list(existing)
    elif n <= len(mining_tiles) and ctx.get("prefer_mining", True):
        chosen = mining_tiles[:n]
    else:
        base = mining_tiles[: min(len(mining_tiles), n)] if ctx.get("prefer_mining", True) else []
        pool = [t for t in all_tiles if t not in base]
        rng.shuffle(pool)
        need = n - len(base)
        chosen = base + pool[: max(0, need)]

    if len(chosen) < n:
        pool = [t for t in all_tiles if t not in chosen]
        rng.shuffle(pool)
        chosen.extend(pool[: n - len(chosen)])

    chosen = chosen[:n]
    rows = []
    for i, tile in enumerate(chosen, 1):
        texts, polys = parse_dota_txt(angle_sweep_ann_path(repo, tile, 0))
        gt_sv = sum(1 for t in texts if is_sv_label(t))
        gt_non_sv = sum(1 for t in texts if is_non_sv_class(t))
        rows.append(dict(
            rank=i, tile_id=tile, gt_total=len(texts), gt_sv_count=gt_sv,
            gt_non_sv_count=gt_non_sv,
            in_mining_500=int(tile in mining_tiles),
            cohort_tag=f"cohort_{n}",
        ))
    write_csv(cohort_path, rows)
    ctx["cohort_tiles"] = chosen
    ctx["cohort_path"] = str(cohort_path)
    return {"status": "OK", "n": len(rows), "path": str(cohort_path)}


def run_r1_infer(ctx: dict) -> dict:
    repo, wd = ctx["repo"], ctx["work"]
    tiles = ctx.get("cohort_tiles") or [r["tile_id"] for r in read_csv(Path(ctx["cohort_path"]))]
    gpu_ids = parse_gpu_id_list(ctx.get("infer_gpu_ids", "4,5,6,7"))
    batch_size = int(ctx.get("infer_batch_size", 48))
    cache_dir = wd / "pred_cache"
    ensure_dir(cache_dir)
    log_path = wd / "logs" / "r1_infer.log"
    ensure_dir(log_path.parent)

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] {msg}\n")

    errors: list[str] = []
    for angle in ANGLES_12:
        pending = [
            t for t in tiles
            if not (
                angle_sweep_pred_cache_path(cache_dir, t, angle).exists()
                and angle_sweep_pred_cache_path(cache_dir, t, angle).stat().st_size > 64
            )
        ]
        if not pending:
            _log(f"angle {angle:03d} skip all cached ({len(tiles)} tiles)")
            continue
        _log(f"angle {angle:03d} infer {len(pending)}/{len(tiles)} batch={batch_size} gpus={gpu_ids}")
        _, errs = run_angle_sweep_inference_parallel(
            repo, pending, angle, gpu_ids, cache_dir, {},
            log_fn=_log, batch_size=batch_size, work_dir=wd,
        )
        errors.extend(errs)

    return {"status": "OK", "n_tiles": len(tiles), "n_angles": len(ANGLES_12), "infer_errors": len(errors)}


def analyze_tile_angle(
    repo: Path, cache_dir: Path, tile: str, angle: int, pred0: dict | None,
) -> tuple[list[dict], list[dict], dict]:
    gt = load_gt(tile, angle, repo)
    pred = load_pred(repo, cache_dir, tile, angle)
    gt0 = load_gt(tile, 0, repo) if angle != 0 else gt
    track = gt_track_map(gt0, gt) if angle != 0 else list(range(len(gt["texts"])))

    events_gt: list[dict] = []
    events_sv: list[dict] = []

    sv_idx = [i for i, t in enumerate(pred["texts"]) if is_sv_label(t)]
    gt_sv_idx = [i for i, t in enumerate(gt["texts"]) if is_sv_label(t)]
    non_sv_gt_idx = [i for i, t in enumerate(gt["texts"]) if is_non_sv_class(t)]

    # Case-1: non-SV GT covered by SV pred
    iou_sv_gt = (
        iou_matrix_rotated(gt["polys"][non_sv_gt_idx], pred["polys"][sv_idx])
        if non_sv_gt_idx and sv_idx else np.zeros((len(non_sv_gt_idx), len(sv_idx)))
    )
    case1_gt = set()
    for ii, gi in enumerate(non_sv_gt_idx):
        if iou_sv_gt.shape[0] <= ii:
            continue
        gcls = gt["texts"][gi]
        if not is_case1_eligible_gt(gcls):
            continue
        best_j = int(iou_sv_gt[ii].argmax()) if iou_sv_gt.shape[1] else -1
        best_iou = float(iou_sv_gt[ii, best_j]) if best_j >= 0 else 0.0
        if best_iou >= IOU_CASE1:
            case1_gt.add(gi)
            pj = sv_idx[best_j]
            events_gt.append(dict(
                tile_id=tile, angle=angle, event="case1", gt_idx=gi,
                gt_class=gcls, pred_idx=pj, iou=best_iou,
                drift_type="sv_on_non_sv_gt",
            ))

    # Case-2 per GT instance
    for gi in range(len(gt["texts"])):
        if gi in case1_gt:
            continue
        gcls = gt["texts"][gi]
        gpoly = gt["polys"][gi]
        if poly_area(gpoly) < MIN_GT_AREA:
            continue
        pj, iou_p, pcls = best_pred_for_gt(gpoly, pred)

        ref_iou = iou_p
        ref_pcls = pcls
        ref_shift = 0.0
        if angle != 0 and pred0 is not None:
            g0_i = track[gi] if gi < len(track) else -1
            if g0_i >= 0 and g0_i < len(gt0["polys"]):
                _, ref_iou, ref_pcls = best_pred_for_gt(gt0["polys"][g0_i], pred0)
                cx0, cy0 = poly_centroid(gt0["polys"][g0_i])
                cx1, cy1 = poly_centroid(gpoly)
                ref_shift = math.hypot(cx1 - cx0, cy1 - cy0)

        drift_type = ""
        if pj < 0 or iou_p < IOU_WEAK:
            if poly_area(gpoly) >= MIN_GT_AREA:
                drift_type = "miss"
        elif is_non_sv_class(gcls) and is_non_sv_class(pcls) and pcls != gcls and iou_p >= IOU_WEAK:
            drift_type = "cls_non_sv"
        elif is_sv_label(gcls) and is_sv_label(pcls) and angle != 0:
            if ref_iou - iou_p > DIoU_DROP or ref_shift > SHIFT_PX:
                drift_type = "loc_sv"
        elif angle != 0 and pred0 is not None:
            if (ref_iou - iou_p) > DIoU_DROP and iou_p >= IOU_WEAK:
                drift_type = "loc"
            elif ref_shift > SHIFT_PX and iou_p >= IOU_WEAK:
                drift_type = "loc"

        if drift_type:
            events_gt.append(dict(
                tile_id=tile, angle=angle, event="case2", gt_idx=gi,
                gt_class=gcls, pred_idx=pj, iou=iou_p, pred_class=pcls,
                ref_iou=ref_iou, shift_px=ref_shift, drift_type=drift_type,
            ))

    # SV pred attribution
    all_gt_polys = gt["polys"]
    for pi in sv_idx:
        if len(all_gt_polys) == 0:
            events_sv.append(dict(
                tile_id=tile, angle=angle, event="sv_bg", pred_idx=pi,
                score=float(pred["scores"][pi]),
            ))
            continue
        ious = iou_matrix_rotated(pred["polys"][pi].reshape(1, 8), all_gt_polys)[0]
        best_g = int(ious.argmax()) if len(ious) else -1
        best_iou = float(ious[best_g]) if best_g >= 0 else 0.0
        gcls = gt["texts"][best_g] if best_g >= 0 else ""
        # Exclude LV↔SV confusion from Case-1 (count as background SV).
        if gcls in CASE1_EXCLUDE_GT_CLASSES and best_iou >= IOU_CASE1:
            tag = "sv_bg"
        elif best_g in case1_gt and best_iou >= IOU_CASE1:
            tag = "sv_case1"
        elif best_iou < IOU_BG:
            tag = "sv_bg"
        else:
            tag = "sv_residual"
        events_sv.append(dict(
            tile_id=tile, angle=angle, event=tag, pred_idx=pi,
            score=float(pred["scores"][pi]), gt_idx=best_g, gt_class=gcls, iou=best_iou,
        ))

    summary = dict(
        tile_id=tile, angle=angle,
        n_gt=len(gt["texts"]), n_gt_sv=len(gt_sv_idx), n_gt_non_sv=len(non_sv_gt_idx),
        n_pred=len(pred["texts"]), n_pred_sv=len(sv_idx),
        n_case1_gt=len(case1_gt), n_case2=sum(1 for e in events_gt if e["event"] == "case2"),
        n_sv_bg=sum(1 for e in events_sv if e["event"] == "sv_bg"),
        n_sv_case1=sum(1 for e in events_sv if e["event"] == "sv_case1"),
    )
    return events_gt, events_sv, summary


def count_pred_cache(wd: Path, tiles: list[str]) -> dict[str, Any]:
    cache_dir = wd / "pred_cache"
    per_angle = {}
    for ang in ANGLES_12:
        suf = f"angle_{ang:03d}_results.pkl"
        per_angle[ang] = sum(
            1 for t in tiles
            if (cache_dir / t / suf).exists()
            and (cache_dir / t / suf).stat().st_size > 64
        )
    total = sum(per_angle.values())
    return dict(
        total=total,
        expected=len(tiles) * len(ANGLES_12),
        per_angle=per_angle,
        n_tiles_with_cache=len([
            t for t in tiles
            if cache_dir.joinpath(t).is_dir()
            and any(cache_dir.joinpath(t).glob("angle_*_results.pkl"))
        ]),
    )


def validate_r1_complete(wd: Path, tiles: list[str], min_ratio: float = 0.995) -> tuple[bool, dict]:
    st = count_pred_cache(wd, tiles)
    ok = st["total"] >= int(st["expected"] * min_ratio)
    return ok, st


def load_cohort_tiles(ctx: dict) -> list[str]:
    if ctx.get("cohort_tiles"):
        return list(ctx["cohort_tiles"])
    path = Path(ctx["cohort_path"])
    if not path.exists():
        raise FileNotFoundError(f"cohort missing: {path}")
    return [r["tile_id"] for r in read_csv(path)]


def _r2_flush(
    wd: Path,
    all_gt_events: list[dict],
    all_sv_events: list[dict],
    summaries: list[dict],
) -> None:
    write_csv(wd / "ftable_case_gt_events.csv", all_gt_events)
    write_csv(wd / "ftable_case_sv_events.csv", all_sv_events)
    write_csv(wd / "ftable_case_summary_tile_angle.csv", summaries)


def _r2_done_tiles(summaries: list[dict], n_angles: int = 12) -> set[str]:
    c = Counter(s["tile_id"] for s in summaries)
    return {t for t, n in c.items() if n >= n_angles}


def validate_r2_complete(
    wd: Path, tiles: list[str], n_angles: int = 12,
) -> tuple[bool, dict[str, Any]]:
    path = wd / "ftable_case_summary_tile_angle.csv"
    if not path.exists() or path.stat().st_size <= 100:
        return False, dict(reason="missing", n_rows=0, n_tiles_done=0, expected=len(tiles))
    rows = read_csv(path)
    done = _r2_done_tiles(rows, n_angles=n_angles)
    ok = len(done) >= len(tiles)
    return ok, dict(
        reason="ok" if ok else "incomplete",
        n_rows=len(rows),
        n_tiles_done=len(done),
        expected=len(tiles),
        expected_rows=len(tiles) * n_angles,
    )


def _r2_process_one_tile(args: tuple[str, str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Worker: one tile × 12 angles (picklable top-level)."""
    repo_s, cache_s, tile = args
    repo = Path(repo_s)
    cache_dir = Path(cache_s)
    pred0 = load_pred(repo, cache_dir, tile, 0)
    all_gt: list[dict] = []
    all_sv: list[dict] = []
    summaries: list[dict] = []
    for angle in ANGLES_12:
        eg, es, sm = analyze_tile_angle(
            repo, cache_dir, tile, angle, pred0 if angle != 0 else None,
        )
        all_gt.extend(eg)
        all_sv.extend(es)
        summaries.append(sm)
    return all_gt, all_sv, summaries


def run_r2_events(ctx: dict) -> dict:
    repo, wd = ctx["repo"], ctx["work"]
    tiles = load_cohort_tiles(ctx)
    cache_dir = wd / "pred_cache"
    log_path = wd / "logs" / "r2_events.log"
    ensure_dir(log_path.parent)

    all_gt_events: list[dict] = []
    all_sv_events: list[dict] = []
    summaries: list[dict] = []
    done_tiles: set[str] = set()

    summary_path = wd / "ftable_case_summary_tile_angle.csv"
    if ctx.get("resume_r2") and summary_path.exists() and summary_path.stat().st_size > 100:
        summaries = read_csv(summary_path)
        done_tiles = _r2_done_tiles(summaries)
        gt_p = wd / "ftable_case_gt_events.csv"
        sv_p = wd / "ftable_case_sv_events.csv"
        if gt_p.exists() and gt_p.stat().st_size > 100:
            all_gt_events = read_csv(gt_p)
        if sv_p.exists() and sv_p.stat().st_size > 100:
            all_sv_events = read_csv(sv_p)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] r2 resume: {len(done_tiles)} tiles already complete\n")

    pending = [t for t in tiles if t not in done_tiles]
    n_workers = max(1, int(ctx.get("r2_workers", 1)))
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{now_iso()}] r2 start pending={len(pending)}/{len(tiles)} "
            f"workers={n_workers}\n",
        )

    repo_s, cache_s = str(repo), str(cache_dir)
    flush_every = max(1, min(25, len(pending) // 20 or 1))

    if n_workers == 1 or len(pending) <= 1:
        for i, tile in enumerate(pending, 1):
            eg, es, sm_list = _r2_process_one_tile((repo_s, cache_s, tile))
            all_gt_events.extend(eg)
            all_sv_events.extend(es)
            summaries.extend(sm_list)
            if i % flush_every == 0 or i == len(pending):
                _r2_flush(wd, all_gt_events, all_sv_events, summaries)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{now_iso()}] r2 progress {len(done_tiles) + i}/{len(tiles)} "
                        f"summaries={len(summaries)}\n",
                    )
    else:
        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_r2_process_one_tile, (repo_s, cache_s, tile)): tile
                for tile in pending
            }
            for fut in as_completed(futures):
                tile = futures[fut]
                try:
                    eg, es, sm_list = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"r2 failed on tile {tile}") from exc
                all_gt_events.extend(eg)
                all_sv_events.extend(es)
                summaries.extend(sm_list)
                done_count += 1
                if done_count % flush_every == 0 or done_count == len(pending):
                    _r2_flush(wd, all_gt_events, all_sv_events, summaries)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"[{now_iso()}] r2 progress "
                            f"{len(done_tiles) + done_count}/{len(tiles)} "
                            f"summaries={len(summaries)} workers={n_workers}\n",
                        )

    _r2_flush(wd, all_gt_events, all_sv_events, summaries)
    ok, st = validate_r2_complete(wd, tiles)
    if not ok:
        raise RuntimeError(
            f"r2 incomplete: {st['n_tiles_done']}/{st['expected']} tiles, "
            f"{st['n_rows']}/{st['expected_rows']} rows",
        )
    if not summaries:
        raise RuntimeError("r2 produced no summary rows")
    return {
        "status": "OK", "n_tiles": len(tiles),
        "n_gt_events": len(all_gt_events), "n_sv_events": len(all_sv_events),
        "n_summaries": len(summaries),
    }


def _case1_sv_fraction_row(s: dict) -> float:
    sv = int(s["n_pred_sv"])
    return int(s["n_sv_case1"]) / sv if sv else 0.0


def _r3_tile_bundle(item: tuple[str, list[dict]]) -> tuple[dict, dict, list[dict]]:
    """Per-tile: summary row, rotation-case1 row, per-angle delta rows."""
    tile, ss = item
    by_ang = {int(s["angle"]): s for s in ss}
    fracs = {a: _case1_sv_fraction_row(by_ang[a]) for a in ANGLES_12 if a in by_ang}

    sv_det = sum(int(s["n_pred_sv"]) for s in ss)
    case1 = sum(int(s["n_sv_case1"]) for s in ss)
    case2 = sum(int(s["n_case2"]) for s in ss)
    bg = sum(int(s["n_sv_bg"]) for s in ss)
    peak = max(ss, key=lambda x: int(x["n_sv_case1"]) / max(int(x["n_pred_sv"]), 1))

    tile_row = dict(
        tile_id=tile,
        angles=len(ss),
        total_sv_dets=sv_det,
        total_sv_case1=case1,
        total_sv_bg=bg,
        total_case2_events=case2,
        case1_sv_fraction=case1 / sv_det if sv_det else 0.0,
        case1_peak_angle=peak["angle"],
        case1_peak_rate=int(peak["n_sv_case1"]) / max(int(peak["n_pred_sv"]), 1),
    )

    f0 = fracs.get(0, 0.0)
    sv_case1_0 = int(by_ang[0]["n_sv_case1"]) if 0 in by_ang else 0
    # Rotation-induced: Case-1 beyond θ=0° reference (det + fraction spread)
    rot_induced_sv = max(0, case1 - sv_case1_0)
    rot_frac_gain_max = (max(fracs.values()) - f0) if fracs else 0.0
    rot_frac_range = (max(fracs.values()) - min(fracs.values())) if fracs else 0.0
    peak_ang = max(fracs, key=fracs.get) if fracs else 0
    elevated = [a for a in ANGLES_12 if a in fracs and fracs[a] > f0 + 0.05]

    rot_row = dict(
        tile_id=tile,
        case1_frac_angle000=f0,
        case1_frac_max=max(fracs.values()) if fracs else 0.0,
        case1_frac_rot_gain_max=rot_frac_gain_max,
        case1_frac_rot_range=rot_frac_range,
        case1_peak_angle=peak_ang,
        n_sv_case1_angle000=sv_case1_0,
        n_sv_case1_total=case1,
        n_sv_case1_rot_induced=rot_induced_sv,
        rot_case1_fraction=rot_induced_sv / case1 if case1 else 0.0,
        n_angles_case1_elevated_vs_0=len(elevated),
        angles_case1_elevated=";".join(f"{a:03d}" for a in elevated),
        is_peak_nonzero=int(peak_ang != 0),
    )

    delta_rows = []
    for a in ANGLES_12:
        if a not in by_ang:
            continue
        s = by_ang[a]
        fa = fracs[a]
        delta_rows.append(dict(
            tile_id=tile,
            angle=a,
            case1_sv_fraction=fa,
            case1_frac_delta_vs_000=fa - f0,
            n_pred_sv=int(s["n_pred_sv"]),
            n_sv_case1=int(s["n_sv_case1"]),
            n_case1_gt=int(s["n_case1_gt"]),
        ))

    return tile_row, rot_row, delta_rows


def run_r3_aggregate(ctx: dict) -> dict:
    wd = ctx["work"]
    summaries = read_csv(wd / "ftable_case_summary_tile_angle.csv")
    if not summaries:
        raise RuntimeError("r3 requires ftable_case_summary_tile_angle.csv from r2")

    by_tile: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_tile[s["tile_id"]].append(s)

    items = list(by_tile.items())
    n_workers = max(1, int(ctx.get("r3_workers", 64)))
    tile_rows: list[dict] = []
    rot_rows: list[dict] = []
    delta_rows: list[dict] = []

    if n_workers == 1 or len(items) < 64:
        for it in items:
            tr, rr, drs = _r3_tile_bundle(it)
            tile_rows.append(tr)
            rot_rows.append(rr)
            delta_rows.extend(drs)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for tr, rr, drs in pool.map(_r3_tile_bundle, items, chunksize=64):
                tile_rows.append(tr)
                rot_rows.append(rr)
                delta_rows.extend(drs)

    tile_rows.sort(key=lambda r: -float(r["case1_sv_fraction"]))
    write_csv(wd / "ftable_case_summary_tile.csv", tile_rows)

    rot_rows.sort(key=lambda r: -float(r["case1_frac_rot_gain_max"]))
    write_csv(wd / "ftable_case1_rotation_tile.csv", rot_rows)

    write_csv(wd / "ftable_case1_rotation_tile_angle.csv", delta_rows)

    # Cohort-level: mean case1 fraction & delta vs 0° per angle
    by_ang_delta: dict[int, list[float]] = defaultdict(list)
    by_ang_frac: dict[int, list[float]] = defaultdict(list)
    for r in delta_rows:
        a = int(r["angle"])
        by_ang_frac[a].append(float(r["case1_sv_fraction"]))
        by_ang_delta[a].append(float(r["case1_frac_delta_vs_000"]))

    ang_cohort_rows = []
    for a in ANGLES_12:
        frs = by_ang_frac.get(a, [])
        dls = by_ang_delta.get(a, [])
        if not frs:
            continue
        ang_cohort_rows.append(dict(
            angle=a,
            n_tiles=len(frs),
            mean_case1_sv_fraction=sum(frs) / len(frs),
            mean_case1_frac_delta_vs_000=sum(dls) / len(dls),
            median_case1_frac_delta_vs_000=sorted(dls)[len(dls) // 2],
            n_tiles_delta_gt_005=sum(1 for d in dls if d > 0.05),
        ))
    write_csv(wd / "ftable_case1_rotation_angle_cohort.csv", ang_cohort_rows)

    by_ang: dict[int, list[dict]] = defaultdict(list)
    for s in summaries:
        by_ang[int(s["angle"])].append(s)

    ang_rows = []
    for ang, ss in sorted(by_ang.items()):
        sv_det = sum(int(s["n_pred_sv"]) for s in ss)
        case1 = sum(int(s["n_sv_case1"]) for s in ss)
        ang_rows.append(dict(
            angle=ang, n_tiles=len(ss), total_sv_dets=sv_det,
            total_sv_case1=case1,
            case1_sv_fraction=case1 / sv_det if sv_det else 0.0,
        ))
    write_csv(wd / "ftable_case_summary_angle.csv", ang_rows)

    rmd = ctx.get("result_md")
    if rmd:
        _write_rotation_case1_report(Path(rmd), rot_rows, ang_cohort_rows)

    return {
        "status": "OK",
        "n_tiles": len(tile_rows),
        "n_angles": len(ang_rows),
        "n_rotation_tile_rows": len(rot_rows),
        "n_rotation_tile_angle_rows": len(delta_rows),
        "r3_workers": n_workers,
    }


def _write_rotation_case1_report(
    rmd: Path,
    rot_rows: list[dict],
    ang_cohort_rows: list[dict],
) -> None:
    top_rot = sorted(rot_rows, key=lambda r: -float(r["case1_frac_rot_gain_max"]))[:15]
    md = [
        "# Rotation-induced Case-1 statistics",
        "",
        f"- generated_at: `{now_iso()}`",
        "",
        "## Definitions",
        "",
        f"- Case-1 excludes GT classes: {sorted(CASE1_EXCLUDE_GT_CLASSES)}.",
        "- `case1_frac_angle000`: Case-1 SV fraction at reference angle 0°.",
        "- `case1_frac_rot_gain_max`: max over 12 angles of (frac@θ − frac@0°).",
        "- `n_sv_case1_rot_induced`: total sv_case1 − sv_case1@0° (detections beyond 0° baseline).",
        "- `rot_case1_fraction`: rot_induced / total sv_case1 on tile.",
        "",
        "## Cohort mean Δ(case1_frac) vs angle 0°",
        "",
        "| angle | mean_frac | mean_Δvs0 | n_tiles_Δ>0.05 |",
        "| --- | --- | --- | --- |",
    ]
    for r in ang_cohort_rows:
        md.append(
            f"| {int(r['angle']):03d} | {float(r['mean_case1_sv_fraction']):.4f} | "
            f"{float(r['mean_case1_frac_delta_vs_000']):+.4f} | {r['n_tiles_delta_gt_005']} |",
        )
    md.extend([
        "",
        "## Top-15 tiles by rotation Case-1 gain (max_Δvs0)",
        "",
        "| tile_id | frac@0 | max_frac | rot_gain | rot_induced_sv | peak_ang |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for r in top_rot:
        md.append(
            f"| {r['tile_id']} | {float(r['case1_frac_angle000']):.3f} | "
            f"{float(r['case1_frac_max']):.3f} | {float(r['case1_frac_rot_gain_max']):.3f} | "
            f"{r['n_sv_case1_rot_induced']} | {int(float(r['case1_peak_angle'])):03d} |",
        )
    (rmd / "fres_013_rotation_case1_stats.md").write_text("\n".join(md), encoding="utf-8")


def run_r4_report(ctx: dict) -> dict:
    rmd, wd = ctx["result_md"], ctx["work"]
    tile_rows = read_csv(wd / "ftable_case_summary_tile.csv")
    if not tile_rows:
        raise RuntimeError("r4 requires ftable_case_summary_tile.csv from r3")

    cohort_n = int(ctx.get("cohort_size", len(tile_rows)))
    md = [
        "# RSV-Taxonomy Summary",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- cohort: n={cohort_n}",
        f"- angles: {ANGLES_12}",
        f"- protocol: angle_sweep co-rotated GT",
        f"- case1_excludes_gt: {sorted(CASE1_EXCLUDE_GT_CLASSES)} "
        "(SV preds on these GT → sv_bg, not Case-1)",
        "",
        "## Top-15 tiles by case1_sv_fraction",
        "",
        "| tile_id | case1_sv_fraction | peak_angle | total_sv_dets | case2_events |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in tile_rows[:15]:
        md.append(
            f"| {r['tile_id']} | {float(r['case1_sv_fraction']):.4f} | "
            f"{int(float(r['case1_peak_angle'])):03d} | {r['total_sv_dets']} | "
            f"{r['total_case2_events']} |",
        )
    out = rmd / "fres_010_rsv_taxonomy_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return {"status": "OK", "n_tiles": len(tile_rows), "report": str(out)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSV rotation Case-1/2 taxonomy suite")
    p.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    p.add_argument("--work-dir", type=Path, default=WORK_DEFAULT)
    p.add_argument("--result-md-dir", type=Path, default=RESULT_MD_DEFAULT)
    p.add_argument("--cohort-size", type=int, default=500)
    p.add_argument("--extend-cohort", action="store_true",
                   help="Extend existing cohort file to cohort-size")
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--infer-gpu-ids", default="4,5,6,7")
    p.add_argument("--infer-batch-size", type=int, default=48)
    p.add_argument("--step", default="all",
                   help="r0,r1,r2,r3,r4,all or comma-separated")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run step even if .done marker exists")
    p.add_argument(
        "--r2-workers",
        type=int,
        default=512,
        help="Parallel workers for r2 tile×angle case mining (default 512)",
    )
    p.add_argument(
        "--r3-workers",
        type=int,
        default=128,
        help="Parallel workers for r3 tile aggregation (default 128)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.work_dir)
    ensure_dir(args.result_md_dir)
    ensure_dir(args.work_dir / "logs")

    ctx = dict(
        repo=args.repo_root.resolve(),
        work=args.work_dir.resolve(),
        result_md=args.result_md_dir.resolve(),
        cohort_size=args.cohort_size,
        extend_cohort=args.extend_cohort,
        seed=args.seed,
        infer_gpu_ids=args.infer_gpu_ids,
        infer_batch_size=args.infer_batch_size,
        cohort_path=str(args.work_dir / f"ftable_cohort_{args.cohort_size}.csv"),
        prefer_mining=True,
        resume_r2=args.resume,
        r2_workers=args.r2_workers,
        r3_workers=args.r3_workers,
    )

    steps = [
        ("r0", run_r0_cohort),
        ("r1", run_r1_infer),
        ("r2", run_r2_events),
        ("r3", run_r3_aggregate),
        ("r4", run_r4_report),
    ]
    want = {s.strip() for s in args.step.split(",")}
    if "all" in want:
        want = {"r0", "r1", "r2", "r3", "r4"}

    log_path = args.work_dir / "logs" / "run_rsv_taxonomy.log"
    results = {}
    for name, fn in steps:
        if name not in want:
            continue
        marker = args.work_dir / f".done_{name}"
        if args.resume and marker.exists() and not args.force:
            if name == "r0":
                n = int(ctx["cohort_size"])
                cp = args.work_dir / f"ftable_cohort_{n}.csv"
                ok = cp.exists() and len(cp.read_text(encoding="utf-8").strip().splitlines()) >= n + 1
                if ok:
                    print(f"skip {name} (resume, cohort_{n}.csv ok)")
                    continue
                print(f"re-run {name} (cohort_{n}.csv missing or short)")
            elif name == "r1":
                tiles = load_cohort_tiles(ctx)
                ok, st = validate_r1_complete(args.work_dir, tiles)
                if ok:
                    print(f"skip {name} (resume, cache {st['total']}/{st['expected']})")
                    continue
                print(f"re-run {name} (marker exists but cache incomplete "
                      f"{st['total']}/{st['expected']})")
            elif name == "r2":
                ok_r2, st_r2 = validate_r2_complete(
                    args.work_dir, load_cohort_tiles(ctx),
                )
                if ok_r2:
                    print(
                        f"skip {name} (resume, {st_r2['n_tiles_done']} tiles, "
                        f"{st_r2['n_rows']} rows)",
                    )
                    continue
                print(
                    f"re-run {name} ({st_r2.get('reason', '?')}: "
                    f"{st_r2.get('n_tiles_done', 0)}/{st_r2.get('expected', '?')} tiles)",
                )
            else:
                print(f"skip {name} (resume)")
                continue
        print(f"run {name}")
        try:
            meta = fn(ctx)
            if name == "r1":
                tiles = load_cohort_tiles(ctx)
                ok, st = validate_r1_complete(args.work_dir, tiles)
                if not ok:
                    raise RuntimeError(
                        f"r1 incomplete: {st['total']}/{st['expected']} pkls")
            if name == "r2":
                tiles = load_cohort_tiles(ctx)
                ok, st = validate_r2_complete(args.work_dir, tiles)
                if not ok:
                    raise RuntimeError(
                        f"r2 incomplete: {st['n_tiles_done']}/{st['expected']} tiles")
            results[name] = meta
            marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            err = traceback.format_exc()
            print(err)
            (args.result_md_dir / f"fres_FAIL_{name}.md").write_text(
                f"# FAIL {name}\n\n```\n{err}\n```\n", encoding="utf-8")
            return 1

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] done {json.dumps(results)}\n")
    print("done:", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

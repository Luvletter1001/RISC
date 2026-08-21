#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from common import PROJECT_ROOT, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_json
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus


FULL_12 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
REPORT_PATH = exp_path("reports", "full_execution_readiness_audit.md")
JSON_PATH = exp_path("reports", "full_execution_readiness_audit.json")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _status_rows() -> dict[str, list[dict]]:
    buckets = {
        "done_smoke": [],
        "done_full": [],
        "smoke_proxy": [],
        "schema_only": [],
    }
    status_to_bucket = {
        ExperimentStatus.DONE_SMOKE: "done_smoke",
        ExperimentStatus.DONE_FULL: "done_full",
        ExperimentStatus.SMOKE_PROXY: "smoke_proxy",
        ExperimentStatus.SCHEMA_ONLY: "schema_only",
    }
    roots = [
        exp_path("outputs", "smoke"),
        exp_path("outputs", "runs"),
        exp_path("outputs", "open_vocab_assets"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            for row in _read_csv(path):
                bucket = status_to_bucket.get(row.get("status", ""))
                if bucket:
                    buckets[bucket].append(
                        {
                            "path": str(path),
                            "model_name": row.get("model_name") or row.get("method") or row.get("smoke") or "",
                            "status": row.get("status", ""),
                            "status_reason": row.get("status_reason", ""),
                        }
                    )
    for manifest_path in sorted(exp_path("outputs").rglob("manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        bucket = status_to_bucket.get(str(manifest.get("status", "")))
        if bucket:
            buckets[bucket].append(
                {
                    "path": str(manifest_path),
                    "model_name": ",".join(manifest.get("selected_models", []) or []),
                    "status": manifest.get("status", ""),
                    "status_reason": manifest.get("status_reason", ""),
                }
            )
    return buckets


def _split_info(name: str) -> dict:
    path = exp_path("outputs", "splits", f"{name}.json")
    if not path.exists():
        return {"exists": False, "path": str(path), "num_tiles": 0, "strata": {}, "ready": False}
    data = read_jsonish(path)
    tiles = data.get("tiles", [])
    tag_counts = Counter()
    missing_files = 0
    for tile in tiles:
        for tag in tile.get("tags", []):
            tag_counts[tag] += 1
        if not Path(tile.get("image_path", "")).exists() or not Path(tile.get("ann_path", "")).exists():
            missing_files += 1
    required_s3 = {"false_sv_hub_candidate", "true_sv_rich", "low_risk_normal", "cross_class_conflict_candidate"}
    strata = dict(sorted(tag_counts.items()))
    required_ok = True
    blocker = ""
    if name == "S3_safety_stratified" and not required_s3.issubset(strata):
        required_ok = False
        blocker = f"missing required S3 strata: {sorted(required_s3 - set(strata))}"
    return {
        "exists": True,
        "path": str(path),
        "num_tiles": int(data.get("num_tiles", len(tiles))),
        "seed": data.get("seed"),
        "strata": strata,
        "missing_source_files": missing_files,
        "ready": bool(tiles) and missing_files == 0 and required_ok,
        "blocker": blocker,
    }


def _closedset_status() -> dict:
    rows = _read_csv(exp_path("outputs", "capability_matrix.csv"))
    available, ready, blocked = [], [], []
    for row in rows:
        if row.get("model_family") != "closed_set":
            continue
        item = {
            "model_name": row.get("model_name", ""),
            "display_name": row.get("display_name", ""),
            "asset_status": row.get("asset_status", ""),
            "asset_status_reason": row.get("asset_status_reason", ""),
        }
        available.append(item)
        if row.get("asset_status") == ExperimentStatus.DONE_SMOKE:
            ready.append(item)
        else:
            blocked.append(item)
    return {"available_models": available, "ready_models": ready, "blocked_models": blocked}


def _open_vocab_status() -> dict:
    inv_path = exp_path("outputs", "open_vocab_assets", "open_vocab_asset_inventory.json")
    available, ready, blocked = [], [], []
    if inv_path.exists():
        inv = read_json(inv_path)
        for pair in inv.get("runnable_pairs", []):
            item = {
                "family": pair.get("candidate_model_family", ""),
                "config": pair.get("config", ""),
                "checkpoint": pair.get("checkpoint", ""),
                "support_pkl": pair.get("support_pkl", ""),
                "usable": bool(pair.get("usable", True)),
                "blocker": pair.get("blocker", ""),
            }
            available.append(item)
            if item["usable"] and item["family"] == "openrsd":
                ready.append(item)
            else:
                blocked.append(item)
    return {"available_models": available, "ready_models": ready, "blocked_models": blocked}


def _existing(path: str) -> bool:
    return bool(path) and Path(path).exists()


def _dehub_status() -> dict:
    baseline_candidates = [
        "/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth",
    ]
    repair_candidates = [
        "/data1/zcy/OpenRSD/work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_3000.pth",
        "/data1/zcy/OpenRSD/work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_5000.pth",
        "/data1/zcy/OpenRSD/work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_8000.pth",
    ]
    baselines = [{"path": p, "exists": _existing(p)} for p in baseline_candidates]
    repairs = [{"path": p, "exists": _existing(p)} for p in repair_candidates]
    ready_pairs, blocked_pairs = [], []
    for baseline in baselines:
        for repair in repairs:
            pair = {"baseline": baseline["path"], "repair": repair["path"]}
            if baseline["exists"] and repair["exists"]:
                ready_pairs.append(pair)
            else:
                pair["blocker"] = "missing_baseline_or_repair_checkpoint"
                blocked_pairs.append(pair)
    return {
        "baseline_checkpoints": baselines,
        "repair_checkpoints": repairs,
        "ready_pairs": ready_pairs,
        "blocked_pairs": blocked_pairs,
    }


def _angle_protocol_status(s2_info: dict) -> dict:
    return {
        "protocol": FULL_12,
        "source_angle_000_ready": bool(s2_info.get("ready")),
        "rotated_assets_generated_on_demand": True,
        "ready": bool(s2_info.get("ready")),
    }


def build_audit() -> dict:
    current = _status_rows()
    splits = {
        "s2_final_test": _split_info("S2_final_test"),
        "s3_safety_stratified": _split_info("S3_safety_stratified"),
    }
    closedset = _closedset_status()
    open_vocab = _open_vocab_status()
    dehub = _dehub_status()
    angle_status = _angle_protocol_status(splits["s2_final_test"])

    immediate, blocked = [], []
    if splits["s2_final_test"]["ready"] and len(closedset["ready_models"]) >= 10 and angle_status["ready"]:
        immediate.append(
            {
                "name": "full_closedset_s2_12angle",
                "output_dir": str(exp_path("outputs", "runs", "full_closedset_s2_12angle")),
                "reason": "S2_final_test, FULL_12, and >=10 checkpoint-backed closed-set models are ready",
            }
        )
    else:
        blocked.append(
            {
                "name": "full_closedset_s2_12angle",
                "blocker": "requires S2_final_test, FULL_12 source data, and >=10 ready closed-set models",
            }
        )
    if splits["s2_final_test"]["ready"] and open_vocab["ready_models"]:
        immediate.append(
            {
                "name": "full_openvocab_s2_12angle_assets_ready",
                "output_dir": str(exp_path("outputs", "runs", "full_openvocab_s2_12angle")),
                "reason": "OpenRSD assets are ready; full runner still needs execution wiring if absent",
            }
        )
    else:
        blocked.append({"name": "full_openvocab_s2_12angle", "blocker": "missing S2 split or ready OpenRSD assets"})
    for name, reason in [
        ("full_causal_intervention_s3_12angle", "requires S3 full predictions and paired intervention runner"),
        ("full_context_counterfactual_s3_12angle", "requires S3 full predictions and condition-level paired outputs"),
        ("full_dehub_safety_s3_12angle", "requires S3 full OpenRSD baseline/repair inference plus true-SV preservation"),
        ("full_per_angle_tta_range", "per-angle TTA runner is not present yet"),
        ("full_oracle_best_view", "depends on per-angle TTA predictions"),
    ]:
        blocked.append({"name": name, "blocker": reason})

    return {
        "current_status": current,
        "splits": splits,
        "angle_protocol": angle_status,
        "closedset": closedset,
        "open_vocab": open_vocab,
        "dehub": dehub,
        "immediate_full_runs": immediate,
        "blocked_full_runs": blocked,
        "source_reports": [
            str(exp_path("reports", "smoke_matrix_report.md")),
            str(exp_path("reports", "openrsd_hook_smoke.md")),
            str(exp_path("reports", "scientific_gap_audit.md")),
            str(exp_path("reports", "capability_matrix.md")),
        ],
    }


def _write_md(audit: dict) -> None:
    def count_status(key: str) -> int:
        return len(audit["current_status"].get(key, []))

    lines = [
        "# Full Execution Readiness Audit",
        "",
        "## Current Status Counts",
        "",
        f"- DONE_SMOKE rows/items: `{count_status('done_smoke')}`",
        f"- DONE_FULL rows/items: `{count_status('done_full')}`",
        f"- SMOKE_PROXY rows/items: `{count_status('smoke_proxy')}`",
        f"- SCHEMA_ONLY rows/items: `{count_status('schema_only')}`",
        "",
        "## Splits",
        "",
    ]
    for key, info in audit["splits"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                f"- exists: `{info['exists']}`",
                f"- path: `{info['path']}`",
                f"- num_tiles: `{info['num_tiles']}`",
                f"- seed: `{info.get('seed', '')}`",
                f"- ready: `{info['ready']}`",
                f"- missing_source_files: `{info.get('missing_source_files', 0)}`",
                f"- strata: `{info.get('strata', {})}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Closed-Set Models",
            "",
            f"- available: `{len(audit['closedset']['available_models'])}`",
            f"- ready: `{len(audit['closedset']['ready_models'])}`",
            f"- blocked: `{len(audit['closedset']['blocked_models'])}`",
            "",
            "| model_name | asset_status | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in audit["closedset"]["available_models"]:
        lines.append(f"| {row['model_name']} | {row['asset_status']} | {row['asset_status_reason']} |")
    lines.extend(
        [
            "",
            "## Open-Vocabulary / OpenRSD Assets",
            "",
            f"- available runnable pairs: `{len(audit['open_vocab']['available_models'])}`",
            f"- ready OpenRSD pairs: `{len(audit['open_vocab']['ready_models'])}`",
            "",
            "## DeHub Assets",
            "",
            f"- baselines: `{audit['dehub']['baseline_checkpoints']}`",
            f"- repairs: `{audit['dehub']['repair_checkpoints']}`",
            f"- ready_pairs: `{len(audit['dehub']['ready_pairs'])}`",
            "",
            "## Immediate Full Runs",
            "",
        ]
    )
    for row in audit["immediate_full_runs"]:
        lines.append(f"- `{row['name']}` -> `{row['output_dir']}`: {row['reason']}")
    lines.extend(["", "## Blocked Full Runs", ""])
    for row in audit["blocked_full_runs"]:
        lines.append(f"- `{row['name']}`: {row['blocker']}")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    audit = build_audit()
    write_json(JSON_PATH, audit)
    _write_md(audit)
    print(f"readiness_json={JSON_PATH}")
    print(f"readiness_md={REPORT_PATH}")


if __name__ == "__main__":
    main()

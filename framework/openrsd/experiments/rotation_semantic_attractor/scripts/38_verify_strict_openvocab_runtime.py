#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/data1/zcy/OpenRSD")
RESULT_DIR = REPO_ROOT / "resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602"
RUN_ROOT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle")
ROWS_CSV = RUN_ROOT / "metrics/open_vocab_benchmark_rows_merged.csv"
INVENTORY = REPO_ROOT / "experiments/rotation_semantic_attractor/outputs/open_vocab_assets/open_vocab_asset_inventory.json"
SPLIT = REPO_ROOT / "experiments/rotation_semantic_attractor/outputs/splits/S2_final_test.json"
CONFIG = REPO_ROOT / "M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
CHECKPOINT = REPO_ROOT / "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"
SUPPORT_PKL = REPO_ROOT / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
PROBE_DIR = REPO_ROOT / "work_dirs/rotation_semantic_attractor_openvocab_strict_probe_20260603"

DOTA1_CLASSES = [
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "ground-track-field",
    "harbor",
    "helicopter",
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
FULL_ANGLES = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330}
EXPECTED_BOOL_TRUE = {
    "has_open_vocab_config",
    "has_checkpoint",
    "has_prompt_or_class_embedding_path",
    "is_embedding_level",
    "is_visual_support_level",
    "reran_inference",
    "actual_inference_run",
    "same_split",
    "same_angles",
    "same_threshold",
    "same_evaluator",
    "has_det_per_img",
    "is_scientific_result",
    "include_in_main_table",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def boolish(value: str) -> bool:
    return str(value).strip().lower() == "true"


def gate(rows: list[dict[str, Any]], name: str, passed: bool, evidence: str, severity: str = "fatal") -> None:
    rows.append(
        {
            "gate": name,
            "status": "PASS" if passed else "FAIL",
            "severity": severity,
            "evidence": evidence,
        }
    )


def load_support_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    classes = list(data.keys()) if isinstance(data, dict) else []
    class_shapes = {}
    missing_embedding_keys = []
    for cls in classes:
        info = data.get(cls, {})
        if not isinstance(info, dict):
            missing_embedding_keys.append(cls)
            continue
        class_shapes[cls] = {}
        for key in ("visual_embeds", "text_embeds"):
            arr = info.get(key)
            if arr is None:
                missing_embedding_keys.append(f"{cls}:{key}")
                continue
            try:
                class_shapes[cls][key] = list(getattr(arr, "shape", []))
            except Exception:
                class_shapes[cls][key] = []
    return {
        "classes": classes,
        "class_count": len(classes),
        "class_order_matches_dota1": classes == DOTA1_CLASSES,
        "missing_embedding_keys": missing_embedding_keys,
        "class_shapes": class_shapes,
    }


def audit_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(ROWS_CSV)
    status_counts = Counter(r.get("status", "") for r in rows)
    angles = sorted({int(float(r.get("angle") or -1)) for r in rows})
    tile_ids = {r.get("tile_id", "") for r in rows}
    key_counts: Counter[tuple[str, str, int, str, str]] = Counter()
    numeric_bad: dict[str, int] = {}
    class_names_seen = set()
    bool_bad: dict[str, int] = defaultdict(int)
    paths = {
        "config": Counter(),
        "checkpoint": Counter(),
        "support_pkl": Counter(),
        "split_name": Counter(),
        "run_family": Counter(),
        "checkpoint_name": Counter(),
        "model_family": Counter(),
    }
    numeric_cols = [
        "detection_total",
        "small_vehicle_count",
        "large_vehicle_count",
        "small_vehicle_ratio",
        "large_vehicle_ratio",
        "mean_score",
        "max_score",
        "num_gt_total",
        "num_gt_sv",
    ]
    for r in rows:
        try:
            angle = int(float(r.get("angle") or -1))
        except Exception:
            angle = -1
        key_counts[(r.get("checkpoint_name", ""), r.get("tile_id", ""), angle, r.get("condition", ""), r.get("intervention", ""))] += 1
        for col in numeric_cols:
            try:
                value = float(r.get(col) or 0)
                if value < 0 or value != value or value in (float("inf"), float("-inf")):
                    numeric_bad[col] = numeric_bad.get(col, 0) + 1
            except Exception:
                numeric_bad[col] = numeric_bad.get(col, 0) + 1
        for col in EXPECTED_BOOL_TRUE:
            if not boolish(r.get(col, "")):
                bool_bad[col] += 1
        for col in paths:
            paths[col][r.get(col, "")] += 1
        try:
            hist = json.loads(r.get("class_histogram") or "{}")
            class_names_seen.update(hist.keys())
        except Exception:
            class_names_seen.add("__INVALID_JSON__")

    duplicates = [k for k, v in key_counts.items() if v > 1]
    per_angle = Counter()
    for r in rows:
        try:
            per_angle[int(float(r.get("angle") or -1))] += 1
        except Exception:
            per_angle[-1] += 1
    summary = {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "angles": angles,
        "unique_tiles": len(tile_ids),
        "duplicate_keys": len(duplicates),
        "duplicate_key_sample": [list(k) for k in duplicates[:10]],
        "per_angle": dict(sorted(per_angle.items())),
        "numeric_bad": numeric_bad,
        "bool_bad": dict(bool_bad),
        "class_names_seen": sorted(class_names_seen),
        "path_modes": {k: paths[k].most_common(5) for k in paths},
    }
    return rows, summary


def audit_manifests() -> dict[str, Any]:
    manifests = sorted(RUN_ROOT.glob("shards/shard_*_of_*/manifest.json"))
    out = {"manifest_paths": [str(p) for p in manifests], "items": []}
    for path in manifests:
        data = json.loads(path.read_text())
        out["items"].append(
            {
                "path": str(path),
                "status": data.get("status"),
                "num_split_tiles": data.get("num_split_tiles"),
                "num_tasks": data.get("num_tasks"),
                "angles": data.get("angles"),
                "device": data.get("args", {}).get("device"),
                "score_thr": data.get("args", {}).get("score_thr"),
                "limit": data.get("args", {}).get("limit"),
                "run_family": data.get("args", {}).get("run_family"),
                "rows_added": data.get("checkpoint_results", [{}])[0].get("rows_added") if data.get("checkpoint_results") else None,
                "failed_chunks_or_tasks": data.get("checkpoint_results", [{}])[0].get("failed_chunks_or_tasks") if data.get("checkpoint_results") else None,
                "storage_policy": data.get("storage_policy"),
            }
        )
    return out


def run_gpu_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_gpu_probe:
        return {"status": "SKIPPED", "reason": "run_gpu_probe_not_requested"}
    out_dir = Path(args.probe_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments/rotation_semantic_attractor/scripts/21_run_full_openrsd_streaming.py"),
        "--split",
        str(SPLIT),
        "--run-family",
        "open_vocab_benchmark",
        "--output-dir",
        str(out_dir),
        "--scratch-dir",
        str(out_dir / "scratch"),
        "--angles",
        "0,30",
        "--limit",
        "2",
        "--chunk-size",
        "2",
        "--device",
        "cuda:0",
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(CHECKPOINT),
        "--support-pkl",
        str(SUPPORT_PKL),
        "--no-resume",
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'tools'}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    (out_dir / "strict_probe_stdout.log").write_text(proc.stdout)
    (out_dir / "strict_probe_stderr.log").write_text(proc.stderr)
    rows_path = out_dir / "metrics/open_vocab_benchmark_rows.csv"
    manifest_path = out_dir / "manifest.json"
    row_count = len(read_csv(rows_path)) if rows_path.exists() else 0
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    logs = sorted((out_dir / "adapter_work/openrsd_a10/runner_work_dir").glob("*.log"))
    log_text = "\n".join(p.read_text(errors="ignore") for p in logs[-2:])
    mismatch_terms = [
        "The model and loaded state dict do not match",
        "missing keys",
        "unexpected key",
        "size mismatch",
    ]
    mismatches = [term for term in mismatch_terms if term.lower() in log_text.lower()]
    combined_output = f"{proc.stdout}\n{proc.stderr}\n{log_text}"
    cuda_unavailable_terms = [
        "CUDA available: False",
        "CUDA is not available",
        "GPU_UNAVAILABLE_IN_TORCH",
    ]
    cuda_unavailable = [term for term in cuda_unavailable_terms if term.lower() in combined_output.lower()]
    return {
        "status": "DONE" if proc.returncode == 0 else "FAILED",
        "returncode": proc.returncode,
        "gpu_id": str(args.gpu_id),
        "command": " ".join(cmd),
        "output_dir": str(out_dir),
        "rows_path": str(rows_path),
        "row_count": row_count,
        "manifest_status": manifest.get("status"),
        "manifest_checkpoint_results": manifest.get("checkpoint_results"),
        "runner_logs": [str(p) for p in logs],
        "state_dict_mismatch_terms": mismatches,
        "cuda_unavailable_terms": cuda_unavailable,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def render_md(gates: list[dict[str, Any]], summary: dict[str, Any], support: dict[str, Any], manifests: dict[str, Any], probe: dict[str, Any]) -> str:
    gate_status = Counter(g["status"] for g in gates)
    fatal_fails = [g for g in gates if g["status"] != "PASS" and g["severity"] == "fatal"]
    verdict = "STRICT_OPENVOCAB_VERIFIED" if not fatal_fails else "STRICT_OPENVOCAB_BLOCKED"
    lines = [
        "# Strict Open-Vocab Implementation / Config / Format Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Gate Summary",
        "",
        f"- PASS: `{gate_status.get('PASS', 0)}`",
        f"- FAIL: `{gate_status.get('FAIL', 0)}`",
        "",
        "| gate | status | severity | evidence |",
        "|---|---|---|---|",
    ]
    for g in gates:
        lines.append(f"| {g['gate']} | {g['status']} | {g['severity']} | {str(g['evidence']).replace('|', '/')} |")
    lines.extend(
        [
            "",
            "## Full Benchmark Format",
            "",
            f"- rows: `{summary['row_count']}`",
            f"- unique tiles: `{summary['unique_tiles']}`",
            f"- angles: `{summary['angles']}`",
            f"- status counts: `{summary['status_counts']}`",
            f"- duplicate keys: `{summary['duplicate_keys']}`",
            f"- class names seen in histograms: `{summary['class_names_seen']}`",
            "",
            "## Support / Class Mapping",
            "",
            f"- support pkl: `{SUPPORT_PKL}`",
            f"- support class count: `{support['class_count']}`",
            f"- support order matches DOTA1 diagnostic order: `{support['class_order_matches_dota1']}`",
            f"- missing embedding keys: `{support['missing_embedding_keys']}`",
            "",
            "## Manifest / Storage Policy",
            "",
            f"- shard manifests: `{len(manifests['items'])}`",
            f"- storage policy: rotated images are intentionally transient and not retained in the full streaming run.",
            "",
            "## GPU Runtime Probe",
            "",
            f"- status: `{probe.get('status')}`",
            f"- GPU: `{probe.get('gpu_id', '')}`",
            f"- output dir: `{probe.get('output_dir', '')}`",
            f"- row count: `{probe.get('row_count', '')}`",
            f"- manifest status: `{probe.get('manifest_status', '')}`",
            f"- state-dict mismatch terms: `{probe.get('state_dict_mismatch_terms', [])}`",
            f"- CUDA unavailable terms: `{probe.get('cuda_unavailable_terms', [])}`",
            "",
            "## Evidence Boundary",
            "",
            "- This audit verifies OpenRSD implementation/config/checkpoint/support loading and the count-level row format used by the full open-vocab diagnostic.",
            "- The full open-vocab benchmark stores merged metric rows and transient asset indexes, not raw box prediction files; therefore this audit does not convert the full run into AP50/mAP evidence.",
            "- The config file contains an 18-class validation list, but the diagnostic runtime explicitly overrides support to DOTA1 15 classes through `probe_rotated_stage_outputs.build_cfg` and `prepare_support`; this audit verifies the 15-class support/order path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=PROBE_DIR)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--run-gpu-probe", action="store_true")
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    gates: list[dict[str, Any]] = []

    for name, path in {
        "full_rows_csv": ROWS_CSV,
        "inventory": INVENTORY,
        "split": SPLIT,
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "support_pkl": SUPPORT_PKL,
    }.items():
        gate(gates, f"{name}_exists", path.exists(), str(path))

    rows, row_summary = audit_rows()
    support_summary = load_support_summary(SUPPORT_PKL)
    manifest_summary = audit_manifests()
    probe_summary = run_gpu_probe(args)

    gate(gates, "row_count_30000", row_summary["row_count"] == 30000, row_summary["row_count"])
    gate(gates, "status_done_full_30000", row_summary["status_counts"].get("DONE_FULL", 0) == 30000, row_summary["status_counts"])
    gate(gates, "angle_set_12", set(row_summary["angles"]) == FULL_ANGLES, row_summary["angles"])
    gate(gates, "unique_tiles_2500", row_summary["unique_tiles"] == 2500, row_summary["unique_tiles"])
    gate(gates, "per_angle_balanced_2500", all(row_summary["per_angle"].get(a) == 2500 for a in FULL_ANGLES), row_summary["per_angle"])
    gate(gates, "duplicate_keys_zero", row_summary["duplicate_keys"] == 0, row_summary["duplicate_key_sample"])
    gate(gates, "numeric_columns_valid", not row_summary["numeric_bad"], row_summary["numeric_bad"])
    gate(gates, "required_bool_flags_true", not row_summary["bool_bad"], row_summary["bool_bad"])
    gate(gates, "class_histogram_subset_dota1", set(row_summary["class_names_seen"]).issubset(set(DOTA1_CLASSES)), row_summary["class_names_seen"])
    gate(gates, "support_class_order_dota1_15", support_summary["class_order_matches_dota1"], support_summary["classes"])
    gate(gates, "support_embedding_keys_present", not support_summary["missing_embedding_keys"], support_summary["missing_embedding_keys"])
    gate(gates, "three_shard_manifests", len(manifest_summary["items"]) == 3, len(manifest_summary["items"]))
    gate(
        gates,
        "manifests_done_full_no_failures",
        all(m.get("status") == "DONE_FULL" and (m.get("failed_chunks_or_tasks") in (0, None)) for m in manifest_summary["items"]),
        manifest_summary["items"],
    )
    gate(
        gates,
        "manifest_score_threshold_0_3",
        all(float(m.get("score_thr") or -1) == 0.3 for m in manifest_summary["items"]),
        [m.get("score_thr") for m in manifest_summary["items"]],
    )
    gate(
        gates,
        "gpu_probe_done",
        probe_summary.get("status") == "DONE" and probe_summary.get("row_count") == 2,
        probe_summary,
    )
    gate(
        gates,
        "gpu_probe_no_state_dict_mismatch",
        probe_summary.get("status") == "DONE" and not probe_summary.get("state_dict_mismatch_terms"),
        probe_summary.get("state_dict_mismatch_terms"),
    )
    gate(
        gates,
        "gpu_probe_cuda_available",
        probe_summary.get("status") == "DONE" and not probe_summary.get("cuda_unavailable_terms"),
        probe_summary.get("cuda_unavailable_terms"),
    )

    payload = {
        "verdict": "STRICT_OPENVOCAB_VERIFIED" if all(g["status"] == "PASS" or g["severity"] != "fatal" for g in gates) else "STRICT_OPENVOCAB_BLOCKED",
        "inputs": {
            "rows_csv": str(ROWS_CSV),
            "inventory": str(INVENTORY),
            "split": str(SPLIT),
            "config": str(CONFIG),
            "checkpoint": str(CHECKPOINT),
            "support_pkl": str(SUPPORT_PKL),
            "config_sha256": sha256_file(CONFIG) if CONFIG.exists() else "",
            "checkpoint_sha256": sha256_file(CHECKPOINT) if CHECKPOINT.exists() else "",
            "support_sha256": sha256_file(SUPPORT_PKL) if SUPPORT_PKL.exists() else "",
        },
        "gates": gates,
        "row_summary": row_summary,
        "support_summary": support_summary,
        "manifest_summary": manifest_summary,
        "probe_summary": probe_summary,
    }

    out_json = args.result_dir / "open_vocab_strict_implementation_config_format_audit.json"
    out_csv = args.result_dir / "open_vocab_strict_implementation_config_format_gates.csv"
    out_md = args.result_dir / "open_vocab_strict_implementation_config_format_audit.md"
    write_json(out_json, payload)
    write_csv(out_csv, gates, ["gate", "status", "severity", "evidence"])
    out_md.write_text(render_md(gates, row_summary, support_summary, manifest_summary, probe_summary))
    print(json.dumps({"verdict": payload["verdict"], "md": str(out_md), "json": str(out_json), "csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

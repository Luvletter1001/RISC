#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


DOTA1_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

OPENRSD_DOTA1_CLASSES = [
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def angle_hint(text: str) -> str:
    hits = []
    for key in ("le90", "le135", "oc"):
        if key in text:
            hits.append(key)
    return ",".join(hits)


def torch_checkpoint_meta(path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(str(path), map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    dataset_meta = meta.get("dataset_meta") if isinstance(meta.get("dataset_meta"), dict) else {}
    classes = (
        meta.get("CLASSES")
        or meta.get("classes")
        or dataset_meta.get("CLASSES")
        or dataset_meta.get("classes")
    )
    if classes is not None:
        classes = list(classes)
    config = str(meta.get("config", ""))
    return {
        "meta_keys": list(meta.keys()),
        "classes": classes,
        "mmdet_version": str(meta.get("mmdet_version", "")),
        "mmcv_version": str(meta.get("mmcv_version", "")),
        "epoch": meta.get("epoch", ""),
        "iter": meta.get("iter", ""),
        "meta_config_angle_hint": angle_hint(config),
        "meta_config_head": config[:800].replace("\n", " "),
    }


def sample_prediction_metadata(run_dir: Path, model_key: str) -> dict[str, Any]:
    model_dir = run_dir / "raw_predictions" / model_key
    files = sorted(model_dir.glob("*/angle_*.json"))
    if not files:
        return {"result_count": 0}
    path = files[0]
    payload = read_json(path)
    meta = payload.get("metadata", {})
    return {
        "result_count": len(files),
        "sample_prediction_json": str(path),
        "sample_config": meta.get("config", ""),
        "sample_checkpoint": meta.get("checkpoint", ""),
    }


def run_init_detector(
    python_bin: str,
    repo_root: Path,
    config: Path,
    checkpoint: Path,
    log_path: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    code = (
        "from mmdet.apis import init_detector; "
        f"init_detector({str(config)!r}, {str(checkpoint)!r}, device='cpu'); "
        "print('INIT_DETECTOR_DONE')"
    )
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'tools'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [python_bin, "-c", code],
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )
    output = proc.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    mismatch = "The model and loaded state dict do not match exactly" in output
    missing = bool(re.search(r"\bmissing keys? in source state_dict\b", output))
    unexpected = bool(re.search(r"\bunexpected key in source state_dict\b", output))
    critical_patterns = [
        "bbox_head_init.",
        "bbox_head_refine.",
        "roi_head.bbox_head.",
        "bbox_head.fc_cls",
        "rbbox_head.fc_cls",
        "retina_cls",
        "retina_reg",
    ]
    benignish_patterns = [
        ".filter",
        ".expanded_bias",
        "bbox_head.moment_transfer",
    ]
    critical = any(pattern in output for pattern in critical_patterns)
    nonexact_benignish = mismatch and not critical and all(
        token in output for token in ["missing keys in source state_dict"]
    ) and not unexpected
    return {
        "load_returncode": proc.returncode,
        "load_ok": proc.returncode == 0,
        "load_mismatch": mismatch,
        "missing_keys_seen": missing,
        "unexpected_keys_seen": unexpected,
        "critical_load_mismatch": critical,
        "nonexact_benignish_load_warning": nonexact_benignish or (mismatch and any(pattern in output for pattern in benignish_patterns) and not critical),
        "load_log": str(log_path),
        "load_log_tail": output[-4000:],
    }


def run_openrsd_runtime_load(
    python_bin: str,
    repo_root: Path,
    model_key: str,
    cfg: dict[str, Any],
    log_path: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    payload = json.dumps(cfg)
    code = f"""
import json
from pathlib import Path
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_adapter import OpenRSDAdapter
cfg = json.loads({payload!r})
adapter = OpenRSDAdapter({model_key!r}, cfg, {str(repo_root)!r})
adapter.load(device='cpu', image_dir={str(repo_root / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images')!r}, out_dir={str(log_path.parent / 'openrsd_runtime_work')!r}, angles=[0])
runtime = adapter.runtime
print('OPENRSD_RUNTIME_LOAD_DONE')
print('support_class_count', len(runtime.name2id))
print('support_classes', ','.join(runtime.name2id.keys()))
print('model_class', runtime.model.__class__.__name__)
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'tools'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [python_bin, "-c", code],
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )
    output = proc.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    mismatch = "The model and loaded state dict do not match exactly" in output
    missing = bool(re.search(r"\bmissing keys? in source state_dict\b", output))
    unexpected = bool(re.search(r"\bunexpected key in source state_dict\b", output))
    support_match = ",".join(OPENRSD_DOTA1_CLASSES) in output
    return {
        "openrsd_runtime_returncode": proc.returncode,
        "openrsd_runtime_load_ok": proc.returncode == 0,
        "openrsd_runtime_load_mismatch": mismatch,
        "openrsd_runtime_missing_keys_seen": missing,
        "openrsd_runtime_unexpected_keys_seen": unexpected,
        "openrsd_runtime_support_classes_match_dota1": support_match,
        "openrsd_expected_class_order": ",".join(OPENRSD_DOTA1_CLASSES),
        "openrsd_runtime_log": str(log_path),
        "openrsd_runtime_log_tail": output[-4000:],
    }


def audit_closed(args: argparse.Namespace, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_models = sorted(
        p.name for p in (args.closedset_run_dir / "raw_predictions").glob("*") if p.is_dir()
    )
    for model_key, cfg in registry["models"].items():
        if cfg.get("model_type") != "closed_set":
            continue
        if result_models and model_key not in result_models:
            continue
        config = Path(cfg.get("config", ""))
        checkpoint = Path(cfg.get("checkpoint", ""))
        row: dict[str, Any] = {
            "family": "closed_set",
            "model_key": model_key,
            "display_name": cfg.get("name", ""),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "config_exists": config.exists(),
            "checkpoint_exists": checkpoint.exists(),
        }
        pred_meta = sample_prediction_metadata(args.closedset_run_dir, model_key)
        row.update(pred_meta)
        row["has_full_12angle_results"] = pred_meta.get("result_count", 0) == 30000
        row["result_config_matches_registry"] = pred_meta.get("sample_config", "") == str(config)
        row["result_checkpoint_matches_registry"] = pred_meta.get("sample_checkpoint", "") == str(checkpoint)
        if config.exists():
            cfg_text = config.read_text(encoding="utf-8", errors="replace")
            row["config_angle_hint"] = angle_hint(cfg_text)
        if checkpoint.exists():
            try:
                meta = torch_checkpoint_meta(checkpoint)
                row.update(meta)
                row["classes_match_dota1"] = meta.get("classes") == DOTA1_CLASSES
            except Exception as exc:
                row["checkpoint_meta_error"] = f"{type(exc).__name__}:{exc}"
        if config.exists() and checkpoint.exists() and not args.metadata_only:
            try:
                load = run_init_detector(
                    args.python_bin,
                    args.repo_root,
                    config,
                    checkpoint,
                    args.output_dir / "logs" / f"{model_key}_init_detector.log",
                    args.timeout_sec,
                )
                row.update(load)
            except subprocess.TimeoutExpired:
                row.update({"load_ok": False, "load_returncode": "TIMEOUT", "load_log": ""})
            except Exception as exc:
                row.update({"load_ok": False, "load_returncode": "EXCEPTION", "load_error": f"{type(exc).__name__}:{exc}"})
        row["audit_status"] = classify_closed(row)
        rows.append(row)
    return rows


def classify_closed(row: dict[str, Any]) -> str:
    if not row.get("config_exists") or not row.get("checkpoint_exists"):
        return "BLOCKED_MISSING_CONFIG_OR_CHECKPOINT"
    if not row.get("has_full_12angle_results"):
        return "NO_FULL_12ANGLE_RESULT"
    if not row.get("result_config_matches_registry") or not row.get("result_checkpoint_matches_registry"):
        return "INVALID_RESULT_METADATA_MISMATCH"
    if row.get("classes") in ("", None):
        class_status = "CLASS_META_ABSENT"
    elif row.get("classes_match_dota1") is False:
        return "INVALID_CLASS_MAPPING"
    else:
        class_status = ""
    if row.get("load_ok") is False:
        return "LOAD_FAILED"
    if row.get("critical_load_mismatch") or (row.get("load_mismatch") and row.get("unexpected_keys_seen")):
        return "INVALID_CRITICAL_LOAD_MISMATCH"
    if row.get("load_mismatch") or row.get("missing_keys_seen") or row.get("unexpected_keys_seen"):
        return "SUSPECT_NONEXACT_LOAD"
    if row.get("load_ok") is True:
        if class_status:
            return f"VALID_LOAD_CLEAN_{class_status}"
        return "VALID_LOAD_CLEAN"
    return "PARTIAL_METADATA_ONLY"


def audit_open_vocab(args: argparse.Namespace, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    open_rows = read_csv_rows(args.openvocab_csv)
    status_counts: dict[str, int] = {}
    for item in open_rows:
        status_counts[item.get("status", "")] = status_counts.get(item.get("status", ""), 0) + 1
    for model_key, cfg in registry["models"].items():
        config = Path(cfg.get("config", ""))
        checkpoint = Path(cfg.get("checkpoint", ""))
        support = Path(cfg.get("support_pkl", ""))
        row = {
            "family": "open_vocab",
            "model_key": model_key,
            "display_name": cfg.get("name", ""),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "support_pkl": str(support),
            "config_exists": config.exists(),
            "checkpoint_exists": checkpoint.exists(),
            "support_exists": support.exists(),
            "openvocab_csv": str(args.openvocab_csv),
            "openvocab_csv_exists": args.openvocab_csv.exists(),
            "openvocab_rows": len(open_rows),
            "openvocab_status_counts": json.dumps(status_counts, ensure_ascii=False, sort_keys=True),
            "has_full_12angle_results": len(open_rows) == 30000 and status_counts.get("DONE_FULL", 0) == 30000,
            "audit_status": "PARTIAL_OPENRSD_LOAD_NOT_YET_VERIFIED",
        }
        if not row["config_exists"] or not row["checkpoint_exists"] or not row["support_exists"]:
            row["audit_status"] = "BLOCKED_MISSING_OPENRSD_ARTIFACT"
        elif not row["has_full_12angle_results"]:
            row["audit_status"] = "NO_FULL_OPENVOCAB_12ANGLE_RESULT"
        elif args.audit_openrsd_runtime:
            try:
                runtime = run_openrsd_runtime_load(
                    args.python_bin,
                    args.repo_root,
                    model_key,
                    cfg,
                    args.output_dir / "logs" / f"{model_key}_openrsd_runtime_load.log",
                    args.openrsd_timeout_sec,
                )
                row.update(runtime)
                row["audit_status"] = classify_openrsd(row)
            except subprocess.TimeoutExpired:
                row.update({"openrsd_runtime_load_ok": False, "openrsd_runtime_returncode": "TIMEOUT"})
                row["audit_status"] = "OPENRSD_RUNTIME_LOAD_TIMEOUT"
            except Exception as exc:
                row.update({"openrsd_runtime_load_ok": False, "openrsd_runtime_returncode": "EXCEPTION", "openrsd_runtime_error": f"{type(exc).__name__}:{exc}"})
                row["audit_status"] = "OPENRSD_RUNTIME_LOAD_FAILED"
        rows.append(row)
    return rows


def classify_openrsd(row: dict[str, Any]) -> str:
    if row.get("openrsd_runtime_load_ok") is not True:
        return "OPENRSD_RUNTIME_LOAD_FAILED"
    if row.get("openrsd_runtime_load_mismatch") or row.get("openrsd_runtime_missing_keys_seen") or row.get("openrsd_runtime_unexpected_keys_seen"):
        return "OPENRSD_RUNTIME_NONEXACT_LOAD"
    if row.get("openrsd_runtime_support_classes_match_dota1") is not True:
        return "OPENRSD_RUNTIME_SUPPORT_CLASS_MISMATCH"
    return "VALID_OPENRSD_RUNTIME_LOAD_CLEAN"


def write_markdown(path: Path, closed_rows: list[dict[str, Any]], open_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Rotation Model Config-Checkpoint Integrity Audit",
        "",
        "This audit checks whether saved 12-angle results can be trusted as coming from the registered config/checkpoint pair.",
        "",
        "## Closed-set status",
        "",
        "| model | status | result count | load mismatch | classes match | log |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in closed_rows:
        lines.append(
            f"| {row.get('model_key')} | {row.get('audit_status')} | {row.get('result_count')} | "
            f"{row.get('load_mismatch', '')} | {row.get('classes_match_dota1', '')} | `{row.get('load_log', '')}` |"
        )
    lines += [
        "",
        "## Open-vocab status",
        "",
        "| model | status | rows | config | checkpoint | support | runtime log |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in open_rows:
        lines.append(
            f"| {row.get('model_key')} | {row.get('audit_status')} | {row.get('openvocab_rows')} | "
            f"{row.get('config_exists')} | {row.get('checkpoint_exists')} | {row.get('support_exists')} | `{row.get('openrsd_runtime_log', '')}` |"
        )
    invalid = [r for r in closed_rows if str(r.get("audit_status", "")).startswith("INVALID") or r.get("audit_status") == "LOAD_FAILED"]
    lines += [
        "",
        "## Immediate interpretation",
        "",
    ]
    if invalid:
        lines.append("The current closed-set 12-angle benchmark is not fully validated. These models must be excluded or rerun before claiming all closed-set results are correctly configured:")
        for row in invalid:
            lines.append(f"- `{row.get('model_key')}`: `{row.get('audit_status')}`")
    else:
        lines.append("No invalid closed-set load mismatch was detected by this audit.")
    lines.append("")
    if any(str(r.get("audit_status", "")).startswith("VALID_OPENRSD") for r in open_rows):
        lines.append("Open-vocab runtime load parity passed for the audited OpenRSD model.")
    else:
        lines.append("Open-vocab runtime load parity is not fully verified; see the open-vocab status table.")
    lines += [
        "",
        "## Root-cause notes",
        "",
        "- `r3det_kfiou` is a critical mismatch, not a visualization threshold issue. The checkpoint contains old-style head keys such as `bbox_head.*`, `refine_head.*`, and `feat_refine_module.*`; the current repository config builds a MMRotate 1.x `RefineSingleStageDetector` expecting `bbox_head_init.*` and `bbox_head_refine.*`. The detection head is therefore not cleanly loaded, and the saved 12-angle predictions must not be treated as valid evidence.",
        "- `redet` is a critical mismatch. The checkpoint contains old ReDet/RoITransformer style keys such as `bbox_head.*` and `rbbox_head.*`; the current config builds a `CascadeRCNN`/`roi_head.bbox_head.*` style model. The current 12-angle predictions must be excluded or regenerated with a matching official implementation.",
        "- `redet_msrr` loads with nonexact generated equivariant buffer warnings (`filter` / `expanded_bias`) but no critical detection-head mismatch was detected in this pass. It should be marked `SUSPECT_NONEXACT_LOAD`, not `VALID_LOAD_CLEAN`, until the official ReDet buffer handling is documented or reproduced.",
        "- `oriented_reppoints` loads with a nonexact `bbox_head.moment_transfer` warning. It is not a critical head replacement mismatch, but it still fails the strict parameter-no-difference criterion.",
        "- `openrsd_a10_flex_rtm_v3_1_formal` builds the OpenRSD runtime, loads `/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth`, prepares 15 support classes in the OpenRSD diagnostic order, and reports no state-dict mismatch in this audit.",
        "",
        "## Current usable model subset",
        "",
        "Strictly clean closed-set models by this audit:",
        "",
    ]
    for row in closed_rows:
        if row.get("audit_status") == "VALID_LOAD_CLEAN":
            lines.append(f"- `{row.get('model_key')}`")
    clean_meta_absent = [r for r in closed_rows if r.get("audit_status") == "VALID_LOAD_CLEAN_CLASS_META_ABSENT"]
    if clean_meta_absent:
        lines += ["", "Clean load but class metadata absent in checkpoint:"]
        for row in clean_meta_absent:
            lines.append(f"- `{row.get('model_key')}`")
    not_clean = [
        r for r in closed_rows
        if r.get("audit_status") not in {"VALID_LOAD_CLEAN", "VALID_LOAD_CLEAN_CLASS_META_ABSENT"}
    ]
    if not_clean:
        lines += ["", "Not clean enough for the requested final claim:"]
        for row in not_clean:
            lines.append(f"- `{row.get('model_key')}`: `{row.get('audit_status')}`")
    valid_open = [r for r in open_rows if str(r.get("audit_status", "")).startswith("VALID_OPENRSD")]
    if valid_open:
        lines += ["", "Open-vocab models with runtime load parity:"]
        for row in valid_open:
            lines.append(f"- `{row.get('model_key')}`")
    lines += [
        "",
        "## Required next actions before claiming all models are correct",
        "",
        "1. Remove `redet` and `r3det_kfiou` from any paper-level validated closed-set model count, or rerun them with a matching official old-MMRotate implementation/checkpoint-conversion path.",
        "2. Decide whether `redet_msrr` generated-filter missing keys are acceptable under the project definition of parameter-no-difference; if not, rerun with an implementation that reports clean state loading.",
        "3. Decide whether `oriented_reppoints` missing `moment_transfer` is acceptable; if not, fix the implementation/config pair or exclude it.",
        "4. If the final goal requires every originally listed closed-set model, build a clean-load reproduction path for `redet` and `r3det_kfiou`, then rerun their full 12-angle inference and regenerate dependent closed-set summaries.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--closed-registry", type=Path, default=Path("experiments/rotation_semantic_attractor/configs/model_registry.yaml"))
    parser.add_argument("--open-registry", type=Path, default=Path("experiments/rotation_semantic_attractor/configs/open_vocab_model_registry.yaml"))
    parser.add_argument("--closedset-run-dir", type=Path, default=Path("experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle"))
    parser.add_argument("--openvocab-csv", type=Path, default=Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor/full_openvocab_s2_12angle/metrics/open_vocab_benchmark_rows_merged.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602"))
    parser.add_argument("--python-bin", default="/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--openrsd-timeout-sec", type=int, default=600)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--audit-openrsd-runtime", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    for name in ("closed_registry", "open_registry", "closedset_run_dir", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.repo_root / value)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    closed_registry = read_json(args.closed_registry)
    open_registry = read_json(args.open_registry)
    closed_rows = audit_closed(args, closed_registry)
    open_rows = audit_open_vocab(args, open_registry)
    all_rows = closed_rows + open_rows
    write_csv(args.output_dir / "model_config_checkpoint_integrity.csv", all_rows)
    write_json(args.output_dir / "model_config_checkpoint_integrity.json", all_rows)
    write_markdown(args.output_dir / "model_config_checkpoint_integrity.md", closed_rows, open_rows)
    print(f"wrote={args.output_dir / 'model_config_checkpoint_integrity.md'}")


if __name__ == "__main__":
    main()

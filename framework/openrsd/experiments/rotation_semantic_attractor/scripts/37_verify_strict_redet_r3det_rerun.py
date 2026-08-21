#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/data1/zcy/OpenRSD")
RUN_DIR = REPO_ROOT / "experiments/rotation_semantic_attractor/outputs/runs/full_closedset_s2_12angle_strict_redet_r3det_20260603"
REGISTRY = (
    REPO_ROOT
    / "resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602"
    / "redet_r3det_strict_rerun_model_registry.json"
)
OUT_DIR = REPO_ROOT / "resultmd/exp_rotation_semantic_attractor/model_integrity_audit_20260602"
ANGLES = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330}
EXPECTED = 30000


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def prediction_keys(paths: list[Path], root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in paths}


def sample_metadata(paths: list[Path], expected_config: str, expected_checkpoint: str, limit: int) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, str]] = []
    for path in paths[:limit]:
        checked += 1
        try:
            data = load_json(path)
            meta = data.get("metadata", {})
        except Exception as exc:
            mismatches.append({"path": str(path), "reason": f"json_error:{type(exc).__name__}:{exc}"})
            continue
        config = str(meta.get("config", ""))
        checkpoint = str(meta.get("checkpoint", ""))
        if config != expected_config or checkpoint != expected_checkpoint:
            mismatches.append(
                {
                    "path": str(path),
                    "reason": "metadata_config_or_checkpoint_mismatch",
                    "config": config,
                    "checkpoint": checkpoint,
                }
            )
    return {"metadata_checked": checked, "metadata_mismatch_count": len(mismatches), "metadata_mismatches": mismatches[:20]}


def manifest_rows(run_dir: Path, model: str, strict_registry: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    problems = []
    for path in sorted((run_dir / "manifests").glob(f"{model}*.json")):
        try:
            data = load_json(path)
        except Exception as exc:
            problems.append(f"{path}:json_error:{type(exc).__name__}:{exc}")
            continue
        args = data.get("args", {})
        angles = set(int(a) for a in data.get("angles", []))
        row = {
            "model": model,
            "manifest": str(path),
            "split_name": data.get("split_name", ""),
            "angles_ok": angles == ANGLES,
            "limit": args.get("limit", ""),
            "model_registry": args.get("model_registry", ""),
            "registry_ok": str(args.get("model_registry", "")) == strict_registry,
            "tile_shard_index": args.get("tile_shard_index", ""),
            "tile_shard_count": args.get("tile_shard_count", ""),
            "device": data.get("device", ""),
            "cuda_visible_devices": data.get("environment", {}).get("cuda_visible_devices", ""),
        }
        rows.append(row)
        if row["split_name"] != "S2_final_test":
            problems.append(f"{path}:split_not_S2_final_test")
        if not row["angles_ok"]:
            problems.append(f"{path}:angles_not_full12")
        if int(row["limit"] or 0) != 0:
            problems.append(f"{path}:limit_nonzero")
        if not row["registry_ok"]:
            problems.append(f"{path}:registry_not_strict")
    return rows, problems


def verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    registry = load_json(args.model_registry)["models"]
    rows = []
    all_manifest_rows = []
    for model in args.models.split(","):
        model = model.strip()
        cfg = registry[model]
        raw_paths = sorted((args.run_dir / "raw_predictions" / model).glob("*/angle_*.json"))
        canon_paths = sorted((args.run_dir / "canonical_predictions" / model).glob("*/angle_*.json"))
        error_paths = sorted((args.run_dir / "raw_predictions" / model).glob("*/*.error.json"))
        raw_keys = prediction_keys(raw_paths, args.run_dir / "raw_predictions" / model)
        canon_keys = prediction_keys(canon_paths, args.run_dir / "canonical_predictions" / model)
        raw_minus_canon = sorted(raw_keys - canon_keys)
        canon_minus_raw = sorted(canon_keys - raw_keys)
        manifests, manifest_problems = manifest_rows(args.run_dir, model, str(args.model_registry))
        all_manifest_rows.extend(manifests)
        metadata = sample_metadata(raw_paths, str(cfg["config"]), str(cfg["checkpoint"]), args.metadata_sample_limit)
        done = len(raw_paths) >= EXPECTED and len(canon_paths) >= EXPECTED and not error_paths
        keysets_ok = not raw_minus_canon and not canon_minus_raw
        strict_ok = done and keysets_ok and not manifest_problems and metadata["metadata_mismatch_count"] == 0
        rows.append(
            {
                "model": model,
                "raw_count": len(raw_paths),
                "canonical_count": len(canon_paths),
                "error_count": len(error_paths),
                "expected_count": EXPECTED,
                "done_full_candidate": done,
                "raw_minus_canonical_count": len(raw_minus_canon),
                "canonical_minus_raw_count": len(canon_minus_raw),
                "raw_minus_canonical_sample": ";".join(raw_minus_canon[:20]),
                "canonical_minus_raw_sample": ";".join(canon_minus_raw[:20]),
                "manifest_count": len(manifests),
                "manifest_problem_count": len(manifest_problems),
                "manifest_problems": ";".join(manifest_problems[:20]),
                "expected_config": cfg["config"],
                "expected_checkpoint": cfg["checkpoint"],
                "metadata_checked": metadata["metadata_checked"],
                "metadata_mismatch_count": metadata["metadata_mismatch_count"],
                "strict_verified_full_candidate": strict_ok,
                "status": "STRICT_VERIFIED_FULL_CANDIDATE" if strict_ok else "RUNNING_OR_INCOMPLETE",
            }
        )
    write_csv(args.output_dir / "redet_r3det_strict_rerun_verification.csv", rows)
    write_csv(args.output_dir / "redet_r3det_strict_rerun_manifest_audit.csv", all_manifest_rows)
    (args.output_dir / "redet_r3det_strict_rerun_verification.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ReDet / R3Det Strict Rerun Verification",
        "",
        f"Run dir: `{args.run_dir}`",
        f"Registry: `{args.model_registry}`",
        "",
        "| model | raw | canonical | errors | manifests | metadata checked | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['raw_count']} | {row['canonical_count']} | {row['error_count']} | "
            f"{row['manifest_count']} | {row['metadata_checked']} | {row['status']} |"
        )
    args.output_dir.joinpath("redet_r3det_strict_rerun_verification.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model-registry", type=Path, default=REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", default="redet,r3det_kfiou")
    parser.add_argument("--metadata-sample-limit", type=int, default=64)
    args = parser.parse_args()
    rows = verify(args)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

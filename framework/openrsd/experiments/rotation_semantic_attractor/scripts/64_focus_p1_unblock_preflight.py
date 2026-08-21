#!/usr/bin/env python3
"""FOCUS-OVD P1 unblock preflight for loss mapping.

This preflight checks whether the codebase now contains a detector focus-loss
branch and whether the P0 verified crop records contain enough spatial metadata
to build region-level detector targets. It keeps exact pre-NMS provenance as a
separate diagnostic requirement.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_EXP_DIR = Path(
    "resultmd/exp_focus_ovd_p1_unblock_loss_mapping_20260609")
DEFAULT_P0_EXP = Path("resultmd/exp_focus_ovd_p0_train_eval_20260609")
DEFAULT_DENSE_HEAD = Path("M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py")
DEFAULT_DETECTOR = Path("M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py")
DEFAULT_FOCUS_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
SUBDIRS = (
    "preflight", "configs", "targets", "provenance", "loss_wiring",
    "one_batch_smoke", "reports", "tables", "figures", "manifests", "logs",
)
LOSS_KEYS = (
    "loss_focus_support_distill",
    "loss_focus_anti",
    "loss_focus_preserve",
    "loss_focus_migration",
    "loss_focus_total",
)


def ensure_tree(exp_dir: Path) -> None:
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def git_commit(repo_root: Path) -> str:
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


def check(name: str, ok: bool, detail: str,
          severity: str = "blocking") -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "severity": severity,
    }


def inspect_loss_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return check("detector_focus_loss_path_wired", False,
                     f"missing {path}")
    text = path.read_text(encoding="utf-8")
    tokens = [
        "focus_losses",
        "def loss_focus_by_feat",
        "anti_attractor_loss(",
        "preserve_loss(",
        "migration_kl_loss(",
        "focus_target_masks",
    ]
    ok = all(token in text for token in tokens) and all(
        key in text for key in LOSS_KEYS)
    return check(
        "detector_focus_loss_path_wired", ok,
        "focus loss config, target masks, and loss keys are present"
        if ok else "missing detector focus-loss branch tokens")


def inspect_detector_batch_injection(path: Path) -> dict[str, Any]:
    if not path.exists():
        return check("detector_batch_target_mask_injection", False,
                     f"missing {path}", severity="warning")
    text = path.read_text(encoding="utf-8")
    ok = "focus_target_masks" in text
    return check(
        "detector_batch_target_mask_injection", ok,
        "detector passes focus_target_masks from batch metadata"
        if ok else "detector does not yet build/pass focus_target_masks",
        severity="warning")


def inspect_p0_spatial_rows(split_path: Path) -> dict[str, Any]:
    if not split_path.exists():
        return check("p0_verified_spatial_metadata_available", False,
                     f"missing {split_path}")
    rows = read_json(split_path).get("rows", [])
    if not rows:
        return check("p0_verified_spatial_metadata_available", False,
                     "p0 train split has no rows")
    with_meta = 0
    with_polygon = 0
    final_stage = 0
    for row in rows[: min(len(rows), 100)]:
        meta_path = Path(row.get("metadata_json", ""))
        if meta_path.exists():
            with_meta += 1
            meta = read_json(meta_path)
            record = meta.get("raw_prediction_record", "")
            if isinstance(record, str) and record:
                record = json.loads(record)
            if isinstance(record, dict) and record.get("polygon"):
                with_polygon += 1
            if isinstance(record, dict) and record.get("stage") == "final":
                final_stage += 1
    ok = with_meta > 0 and with_polygon > 0
    detail = (
        f"sampled={min(len(rows), 100)} with_meta={with_meta} "
        f"with_polygon={with_polygon} final_stage={final_stage}")
    return check("p0_verified_spatial_metadata_available", ok, detail)


def inspect_declip(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return check("declip_support_disabled", False, f"missing {config_path}")
    text = config_path.read_text(encoding="utf-8")
    bad = "use_declip_support=True" in text
    return check("declip_support_disabled", not bad,
                 "DeCLIP support is not enabled"
                 if not bad else "DeCLIP support appears enabled")


def inspect_scripts(repo_root: Path) -> list[dict[str, Any]]:
    out = []
    expected = {
        65: "65_focus*.py",
        66: "66_focus*.py",
        67: "67_focus*.py",
        68: "68_focus*.py",
        69: "69_build_focus*.py",
    }
    scripts_dir = repo_root / "experiments/rotation_semantic_attractor/scripts"
    for idx, pattern in expected.items():
        matches = sorted(scripts_dir.glob(pattern))
        out.append(check(f"script_{idx}_present", bool(matches),
                         str(matches[0]) if matches else "missing"))
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS P1 Unblock Preflight",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- detector_focus_loss_path_wired: `{payload['detector_focus_loss_path_wired']}`",
        f"- exact_pre_nms_required_for_p1b: `true`",
        "",
        "## Checks",
        "",
        "| check | ok | severity | detail |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["checks"]:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(
            f"| {row['name']} | {row['ok']} | {row['severity']} | `{detail}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This preflight can unblock P1A spatial-region loss mapping only. "
        "P1B exact pre-NMS provenance remains a separate requirement and must "
        "report its own coverage.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--p0-exp", type=Path, default=DEFAULT_P0_EXP)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else repo / args.exp_dir
    p0_exp = args.p0_exp if args.p0_exp.is_absolute() else repo / args.p0_exp
    ensure_tree(exp_dir)

    dense_head = repo / DEFAULT_DENSE_HEAD
    detector = repo / DEFAULT_DETECTOR
    focus_config = repo / DEFAULT_FOCUS_CONFIG
    train_split = p0_exp / "configs/p0_train_split.json"
    checks = [
        check("p0_exp_exists", p0_exp.exists(), str(p0_exp)),
        check("p0_train_split_exists", train_split.exists(), str(train_split)),
        inspect_loss_path(dense_head),
        inspect_detector_batch_injection(detector),
        inspect_p0_spatial_rows(train_split),
        inspect_declip(focus_config),
    ]
    checks.extend(inspect_scripts(repo))
    blocking_failed = [
        row["name"] for row in checks
        if not row["ok"] and row["severity"] == "blocking"
    ]
    loss_wired = next(
        row for row in checks
        if row["name"] == "detector_focus_loss_path_wired")["ok"]
    status = (
        "PASS_P1A_UNBLOCK_PREFLIGHT"
        if not blocking_failed else "BLOCKED_P1A_UNBLOCK_PREFLIGHT")
    payload = {
        "status": status,
        "git_commit": git_commit(repo),
        "exp_dir": str(exp_dir),
        "p0_exp": str(p0_exp),
        "loss_keys": list(LOSS_KEYS),
        "checks": checks,
        "blocking_failed": blocking_failed,
        "detector_focus_loss_path_wired": bool(loss_wired),
        "evidence_level": "STATIC_LOSS_PATH_AND_P0_SPATIAL_METADATA",
    }
    write_json(exp_dir / "preflight/p1_unblock_preflight.json", payload)
    write_json(exp_dir / "manifest.json", {
        "status": status,
        "git_commit": payload["git_commit"],
        "exp_dir": str(exp_dir),
        "p0_exp": str(p0_exp),
        "loss_keys": list(LOSS_KEYS),
    })
    write_json(exp_dir / "manifests/p1_unblock_manifest.json", payload)
    write_markdown(exp_dir / "preflight/p1_unblock_preflight.md", payload)
    print(json.dumps({
        "status": status,
        "blocking_failed": blocking_failed,
        "detector_focus_loss_path_wired": bool(loss_wired),
    }, indent=2))
    return 0 if status == "PASS_P1A_UNBLOCK_PREFLIGHT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

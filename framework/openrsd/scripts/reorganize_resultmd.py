#!/usr/bin/env python3
"""Reorganize resultmd/: per-exp reports/, legacy root files -> exp_*, symlinks at old paths.

Usage:
  python scripts/reorganize_resultmd.py              # full (root buckets + leaf dirs)
  python scripts/reorganize_resultmd.py --root-only  # only resultmd/ loose files
  python scripts/reorganize_resultmd.py --leaf-only  # only nested mixed leaf dirs
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path("/data1/zcy/OpenRSD")
RESULTMD = REPO / "resultmd"

# Root files -> (target_exp_dir, subdir under exp: reports|figures|data)
ROOT_MIGRATIONS: list[tuple[str, str, str]] = []

# exp_ovd* and exp_sstrain* files (misnamed as files, not dirs)
OVD_SSTRAIN = [
    ("exp_ovd0_preflight_openrsd_inventory", ["exp_ovd0_preflight_openrsd_inventory.md"]),
    ("exp_ovd1_zero_shot_prompt_rotation_curve", ["exp_ovd1_zero_shot_prompt_rotation_curve.md"]),
    (
        "exp_ovd2_prompt_engineering_rotation_sensitivity",
        [
            "exp_ovd2_prompt_engineering_rotation_sensitivity.md",
            "exp_ovd2_prompt_engineering_rotation_sensitivity_ap50_curve.csv",
            "exp_ovd2_prompt_engineering_rotation_sensitivity_ap50_curve.pdf",
            "exp_ovd2_prompt_engineering_rotation_sensitivity_ap50_curve.png",
            "exp_ovd2_prompt_engineering_rotation_sensitivity_ap50_curve_table.md",
        ],
    ),
    (
        "exp_ovd3_alignment_vs_fusion_head_rotation",
        [
            "exp_ovd3_alignment_vs_fusion_head_rotation.md",
            "exp_ovd3_alignment_vs_fusion_head_rotation.ap50_line.png",
            "exp_ovd3_alignment_vs_fusion_head_rotation.ap50_line.svg",
        ],
    ),
    ("exp_ovd4_prompt_ensemble_and_rotation_tta_repair", ["exp_ovd4_prompt_ensemble_and_rotation_tta_repair.md"]),
    ("exp_ovd5_open_vocab_cross_view_diagnostic", ["exp_ovd5_open_vocab_cross_view_diagnostic.md"]),
    ("exp_sstrain0_inventory_and_leakage_audit", ["exp_sstrain0_inventory_and_leakage_audit.md"]),
    ("exp_sstrain1_smoke_test", ["exp_sstrain1_smoke_test.md"]),
    ("exp_sstrain2_multigpu_batchsize", ["exp_sstrain2_multigpu_batchsize.md"]),
]

LEGACY_BUCKETS: list[tuple[str, list[str]]] = [
    (
        "exp_legacy_dota1_rotation_20260506",
        [
            "dota1_angle12_combined_h2rbox_h2rboxv2_rtmdetl_20260506.md",
            "dota1_angle12_redet_eval_4gpu_20260506_144306.md",
            "dota1_exp_a_rotation_tta_merge_eval_canonical_20260507.md",
            "dota1_rotation_study_master_summary_20260507.md",
            "dota1_rotation_tta_full_angle_repair_curve_20260507.md",
            "exp1_dota1_full_angle_tta_repair_curve.md",
            "exp2_far1m_angle_sweep_cross_dataset.md",
            "exp3_rtmdet_l_tta_merge_ablation.md",
            "exp4_classwise_geometrywise_breakdown.md",
            "exp5_dota1_rotation_aug_short_finetune.md",
            "exp6_cross_view_consistency_regularization_prototype.md",
            "exp7_feature_cosine_curve.pdf",
            "exp7_feature_cosine_curve.png",
            "exp7_feature_logit_equivariance_diagnostic.md",
            "exp7_feature_logit_equivariance_line_charts.csv",
            "exp7_feature_logit_equivariance_line_charts.pdf",
            "exp7_feature_logit_equivariance_line_charts.png",
            "exp8_far1m_quick_transfer_rotation_stress.md",
            "gpu67_retinanet_vs_gpu012345_rtmdetl_angle12_map_curve.csv",
            "gpu67_retinanet_vs_gpu012345_rtmdetl_angle12_map_curve.pdf",
            "gpu67_retinanet_vs_gpu012345_rtmdetl_angle12_map_curve.png",
            "h2rbox_dota1_angle12_eval_4gpu_20260506_165725.md",
            "h2rbox_dota1_angle12_eval_4gpu_20260506_211453.md",
            "h2rbox_v2_dota1_angle12_eval_4gpu_20260506_223851.md",
            "remote_sensing_noun_phrase_angle12_ap50_curve.csv",
            "remote_sensing_noun_phrase_angle12_ap50_curve.pdf",
            "remote_sensing_noun_phrase_angle12_ap50_curve.png",
            "remote_sensing_noun_phrase_angle12_ap50_curve_table.md",
            "retinanet_msrr_epoch12_dota1_angle12_gpu67_bs96_20260513_210612.md",
            "retinanet_msrr_epoch12_gpu67_angle12_map_curve.pdf",
            "retinanet_msrr_epoch12_gpu67_angle12_map_curve.png",
            "retinanet_msrr_epoch12_gpu67_angle12_map_curve_summary.csv",
            "rotated_retinanet_r50_amp_dota1_angle12_eval_4gpu_20260506_233545.md",
            "rotated_retinanet_r50_msrr_dota1_angle12_eval_4gpu_20260506_235637.md",
            "rotated_rtmdet_l_dota1_angle12_eval_4gpu_20260506_214641.md",
            "rtmdet_l_epoch36_dota1_angle12_gpu012345_bs16_20260513_214115.md",
            "rtmdet_l_epoch36_dota1_angle12_gpu012345_bs32_20260513_214542.md",
            "train_retinanet_amp_ss_train.md",
            "train_retinanet_msrr_ss_train.md",
            "train_rtmdet_l_ss_train.md",
        ],
    ),
    (
        "exp_legacy_openrsd_followup_20260508_203934",
        [f for f in os.listdir(RESULTMD) if f.startswith("20260508_203934_")],
    ),
    (
        "exp_legacy_openrsd_followup_20260509_200920",
        [f for f in os.listdir(RESULTMD) if f.startswith("20260509_200920_")],
    ),
    (
        "exp_legacy_openrsd_followup_20260509_230014",
        [f for f in os.listdir(RESULTMD) if f.startswith("20260509_230014_")],
    ),
    (
        "exp_legacy_preflight",
        [
            "preflight_dryrun.md",
            "preflight_gpu_batch2_dryrun.md",
            "preflight_gpu_batch2_multigpu_batchsize.md",
            "preflight_gpu_batch2_smoke.md",
            "preflight_multigpu_batchsize.md",
            "preflight_ovd_multigpu_batchsize.md",
            "preflight_ovd_smoke.md",
            "preflight_smoke.md",
            "eval_debug_start.md",
        ],
    ),
    (
        "exp_resultmd_global_summaries",
        [
            "summary_all_experiments_dedup.md",
            "summary_four_experiments_20260507.md",
            "summary_gpu_batch2_20260507.md",
            "summary_openrsd_ovd_rotation_20260508.md",
            "summary_rotation_repair_and_four_experiments_20260508.md",
            "summary_sstrain_retrain_12angle_no_tta.md",
            "openrsd_followup_fix_20260509_143407.md",
            "messdet_dota1_train_gpu67_after_rtmdet.md",
            "monitor_sstrain_training_status.md",
        ],
    ),
]

EXP_ROOT_KEEP = frozenset(
    {"README.md", "readme.md", "MIGRATION_MAP.md", "INDEX.md", ".gitkeep"}
)

# Do not normalize inside these experiment roots (already archived layouts).
SKIP_EXP_NORMALIZE = frozenset(
    {
        "exp_autonomous_rotation_semantic_drift_20260521_overnight",
        "exp_sv_shift_400plus_overnight_gpu89_20260522",
        "exp_sv_shift_counterfactual_accounting_gpu67",
        "exp_sv_shift_causal_mechanism_full_gpu89",
        "exp_formal_success_check_gpu89",
        "exp_mechanism_sv_attractor_gpu89",
        "exp_sv_dehub_lite_train_gpu89",
        "exp_official_step123_sv_attractor_eval",
        "exp_sv_attractor_repair_gpu89",
        "exp_sv_attractor_repair_full_20260519",
        "exp_rotation_causal_probe_P0148_gpu89",
        "exp_rotation_final_causal_proof_P0148_gpu89",
        "exp_rotation_overnight_gpu89",
        "exp_rotation_semantic_drift_overnight_gpu89_20260521",
        "exp_rotation_stage_probe_P0148",
        "exp_verify_sv_attractor_gpu89",
        "exp_next_plan_sv_dehub_step23_overnight",
        "exp_full_experiment_report",
        "exp_legacy_dota1_rotation_20260506",
        "exp_legacy_openrsd_followup_20260508_203934",
        "exp_legacy_openrsd_followup_20260509_200920",
        "exp_legacy_openrsd_followup_20260509_230014",
        "exp_legacy_preflight",
        "exp_resultmd_global_summaries",
        "exp_ovd0_preflight_openrsd_inventory",
        "exp_ovd1_zero_shot_prompt_rotation_curve",
        "exp_ovd2_prompt_engineering_rotation_sensitivity",
        "exp_ovd3_alignment_vs_fusion_head_rotation",
        "exp_ovd4_prompt_ensemble_and_rotation_tta_repair",
        "exp_ovd5_open_vocab_cross_view_diagnostic",
        "exp_sstrain0_inventory_and_leakage_audit",
        "exp_sstrain1_smoke_test",
        "exp_sstrain2_multigpu_batchsize",
    }
)

# Nested dirs with flat md/csv/png mixes (not full exp archives).
LEAF_DIRS_TO_ORGANIZE = [
    RESULTMD / "exp_sv_dehub_lite_train_gpu89" / "verify_ap_ablation_20260521",
    RESULTMD / "exp_sv_dehub_lite_train_gpu89" / "verify_ap_ablation_plan3h_20260521",
    RESULTMD / "exp_mechanism_sv_attractor_gpu89" / "archive_completed_20260520_211635",
]

LEAF_SUBDIRS = ("reports", "figures", "data", "logs")

REPORT_NAME_PREFIXES = (
    "fres_",
    "flog_",
    "faudit_",
    "fplan_",
    "ftrace_",
    "ftable_",
    "log_",
)
REPORT_NAME_SUFFIXES = (".md", ".txt")
DATA_SUFFIXES = (".csv", ".tsv", ".json")
FIGURE_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp")


def subdir_for_file(name: str) -> str:
    low = name.lower()
    if low.startswith("log_") and low.endswith(".txt"):
        return "logs"
    if low.endswith(DATA_SUFFIXES) or low.startswith("ftable_") or low.startswith("fmeta_"):
        return "data"
    if low.endswith(FIGURE_SUFFIXES):
        return "figures"
    if low.endswith(REPORT_NAME_SUFFIXES) or low.startswith(REPORT_NAME_PREFIXES):
        return "reports"
    if low.endswith(".md"):
        return "reports"
    return "data"


def is_archived_exp(exp_dir: Path) -> bool:
    if exp_dir.name in SKIP_EXP_NORMALIZE:
        return True
    if (exp_dir / "job_outputs").is_dir():
        return True
    return False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path, log: list[str]) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        if src.resolve() == dst.resolve():
            return
        raise FileExistsError(f"destination exists: {dst}")
    shutil.move(str(src), str(dst))
    log.append(f"MOVE {src.relative_to(RESULTMD)} -> {dst.relative_to(RESULTMD)}")


def symlink_at(old: Path, target: Path, log: list[str]) -> None:
    if old.exists():
        return
    ensure_dir(old.parent)
    rel = os.path.relpath(target, old.parent)
    os.symlink(rel, old)
    log.append(f"LINK {old.relative_to(RESULTMD)} -> {rel}")


def normalize_mixed_dir(base: Path, log: list[str], compat_links: bool = True) -> None:
    """Move flat md/csv/png/log files into reports|figures|data|logs under base."""
    if not base.is_dir():
        return
    for sub in LEAF_SUBDIRS:
        ensure_dir(base / sub)
    for item in list(base.iterdir()):
        if not item.is_file() or item.is_symlink():
            continue
        if item.name in EXP_ROOT_KEEP:
            continue
        sub = subdir_for_file(item.name)
        dst = base / sub / item.name
        if item.resolve() == dst.resolve():
            continue
        move_file(item, dst, log)
        if compat_links:
            symlink_at(item, dst, log)


def normalize_exp_dir(exp_dir: Path, log: list[str]) -> None:
    if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
        return
    if is_archived_exp(exp_dir):
        return
    for item in list(exp_dir.iterdir()):
        if not item.is_file() or item.is_symlink():
            continue
        if item.name in EXP_ROOT_KEEP:
            continue
        sub = subdir_for_file(item.name)
        ensure_dir(exp_dir / sub)
        dst = exp_dir / sub / item.name
        if item.resolve() == dst.resolve():
            continue
        move_file(item, dst, log)
        symlink_at(item, dst, log)


def migrate_named_files(exp_name: str, filenames: list[str], log: list[str]) -> None:
    exp_dir = RESULTMD / exp_name
    ensure_dir(exp_dir)
    for sub in ("reports", "figures", "data", "logs"):
        ensure_dir(exp_dir / sub)
    for fn in filenames:
        src = RESULTMD / fn
        if not src.is_file() or src.is_symlink():
            continue
        sub = subdir_for_file(fn)
        dst = exp_dir / sub / fn
        old_at_root = RESULTMD / fn
        move_file(src, dst, log)
        symlink_at(old_at_root, dst, log)


def migrate_root_loose_files(log: list[str]) -> None:
    for exp_name, files in OVD_SSTRAIN:
        migrate_named_files(exp_name, files, log)
    for exp_name, files in LEGACY_BUCKETS:
        migrate_named_files(exp_name, sorted(set(files)), log)


def write_index(ts: str) -> None:
    lines = [
        "# resultmd index",
        "",
        f"Updated: {ts}",
        "",
        "Canonical layout: `exp_<name>/{reports,figures,data,logs}/`.",
        "Root `*.md` / `*.csv` / `*.png` paths are compatibility symlinks only.",
        "",
        "## Experiments",
        "",
        "| Directory | Reports |",
        "| --- | --- |",
    ]
    for exp_dir in sorted(RESULTMD.glob("exp_*/")):
        if not exp_dir.is_dir():
            continue
        rep = exp_dir / "reports"
        link = f"[{exp_dir.name}]({exp_dir.name}/)"
        if rep.is_dir():
            summaries = sorted(rep.glob("fres_*.md"))[:3]
            hint = ", ".join(s.name for s in summaries) if summaries else "—"
            lines.append(f"| {link} | {hint} |")
        else:
            lines.append(f"| {link} | (no reports/) |")
    (RESULTMD / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize resultmd experiment files")
    parser.add_argument("--root-only", action="store_true", help="Only migrate resultmd/ root loose files")
    parser.add_argument("--leaf-only", action="store_true", help="Only organize nested mixed leaf dirs")
    args = parser.parse_args()

    log: list[str] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.leaf_only:
        migrate_root_loose_files(log)
    if args.leaf_only:
        for leaf in LEAF_DIRS_TO_ORGANIZE:
            normalize_mixed_dir(leaf, log)
    elif not args.root_only:
        for leaf in LEAF_DIRS_TO_ORGANIZE:
            normalize_mixed_dir(leaf, log)
        for exp_dir in sorted(RESULTMD.glob("exp_*/")):
            if exp_dir.is_dir():
                normalize_exp_dir(exp_dir, log)

    # 4) Write migration log
    map_path = RESULTMD / "MIGRATION_MAP.md"
    lines = [
        "# resultmd migration map",
        "",
        f"Generated: {ts}",
        "",
        "Old root paths are symlinks where noted. Prefer `exp_*/reports/` for new records.",
        "",
        "| action | path |",
        "| --- | --- |",
    ]
    for entry in log:
        lines.append(f"| {entry.split(' ', 1)[0]} | `{entry.split(' ', 1)[1]}` |")
    map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5) Root README
    readme = RESULTMD / "README.md"
    if not readme.exists():
        readme.write_text(
            """# resultmd — experiment records

Per-experiment directories only (`exp_<name>/`). Do not add new experiment Markdown at this root.

## Layout (per experiment)

| Subdir | Contents |
|--------|----------|
| `reports/` | `fres_*`, `flog_*`, `faudit_*`, process/final Markdown |
| `figures/` | PNG, PDF, SVG plots |
| `data/` | CSV / JSON tables (`ftable_*`, curves) |
| `scripts/` | Runnable pipelines (when present) |
| `work/` | Checkpoints, runner artifacts (heavy; optional) |

## Index

- **Migration map** (root cleanup): [MIGRATION_MAP.md](MIGRATION_MAP.md)
- **Global summaries**: [exp_resultmd_global_summaries/reports/](exp_resultmd_global_summaries/reports/)
- **Open-vocab (legacy OVD0–5)**: `exp_ovd0_*` … `exp_ovd5_*`
- **SSTRain preflight**: `exp_sstrain0_*` … `exp_sstrain2_*`
- **DOTA1 rotation (20260506–07)**: [exp_legacy_dota1_rotation_20260506/](exp_legacy_dota1_rotation_20260506/)
- **OpenRSD follow-up batches**: `exp_legacy_openrsd_followup_20260508_203934`, `_200920`, `_230014`

## Active GPU89 / mechanism experiments (examples)

| Directory | Topic |
|-----------|--------|
| `exp_sv_shift_causal_mechanism_full_gpu89/` | SV shift causal mechanism |
| `exp_mechanism_sv_attractor_gpu89/` | SV attractor mechanism atlas |
| `exp_official_step123_sv_attractor_eval/` | Official step1–3 eval |
| `exp_autonomous_rotation_semantic_drift_20260521_overnight/` | Autonomous rotation drift |

Reorganize again: `python scripts/reorganize_resultmd.py`
""",
            encoding="utf-8",
        )

    if not args.root_only and not args.leaf_only:
        # Symlinks at exp root -> reports|data|figures|logs (idempotent repair)
        for exp_dir in sorted(RESULTMD.glob("exp_*/")):
            if is_archived_exp(exp_dir):
                continue
            for sub in LEAF_SUBDIRS:
                subdir = exp_dir / sub
                if not subdir.is_dir():
                    continue
                for f in subdir.iterdir():
                    if not f.is_file():
                        continue
                    link = exp_dir / f.name
                    if link.exists():
                        continue
                    symlink_at(link, f, log)

    write_index(ts)

    print(f"Done: {len(log)} operations")
    for line in log[-20:]:
        print(line)
    if len(log) > 20:
        print(f"... ({len(log) - 20} more in MIGRATION_MAP.md)")


if __name__ == "__main__":
    main()

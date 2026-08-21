#!/usr/bin/env python3
"""Generate FOCUS-EQText short-run variant configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from focus_eqtext_common import (  # noqa: E402
    BASE_CONFIG,
    EXP_DIR,
    SPATIAL_TARGETS,
    TRAIN_VARIANTS,
    ensure_exp_tree,
    md_table,
    resolve,
    variant_defs,
    write_csv,
    write_json,
)


def py_literal(value: Any) -> str:
    return repr(value)


def write_py_config(path: Path, variant: dict[str, Any], exp_dir: Path) -> None:
    audit_path = exp_dir / "train" / variant["variant_id"] / "trainable_audit.json"
    base_config = variant.get("base_config", BASE_CONFIG)
    lines = [
        f"_base_ = {str(base_config)!r}",
        "",
        f"work_dir = {str(Path('work_dirs') / 'focus_eqtext_dota_short_20260609' / variant['variant_id'])!r}",
        "num_gpus = 2",
        "batch_size = 1",
        "max_iter_per_epoch = 1000",
        f"trainable_parameters = {py_literal(variant['trainable_substrings'])}",
        "",
        "train_dataloader = dict(",
        "    batch_size=batch_size,",
        "    sampler=dict(",
        "        batch_size=batch_size,",
        "        num_gpus=num_gpus,",
        "        max_iter_per_epoch=max_iter_per_epoch),",
        ")",
        "",
        "model = dict(",
        f"    support_type={variant['support_type']!r},",
        "    use_declip_support=False,",
        "    with_image_rec_losses=False,",
        "    with_aux_bbox_head=False,",
        "    bbox_head=dict(",
        f"        use_focus_ovd={bool(variant['use_focus_ovd'])!r},",
        f"        focus_ovd={py_literal(variant['focus_ovd'])},",
        f"        focus_losses={py_literal(variant['focus_losses'])},",
        "    ),",
        ")",
        "",
        "custom_hooks = [",
        "    dict(",
        "        type='FocusOVDTrainableAuditHook',",
        f"        trainable_substrings={py_literal(variant['trainable_substrings'] or ['NO_TRAINABLE_FOR_EVAL_ONLY'])},",
        f"        log_path={str(audit_path)!r},",
        "        fail_on_unexpected=True,",
        "        priority='VERY_HIGH'),",
        "]",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant_id", "support_type", "profile", "use_focus_ovd",
        "eqtext_enabled", "dual_enabled", "train_required",
        "target_mapping_mode",
    ]
    lines = ["# FOCUS-EQText Generated Configs", ""]
    lines.extend(md_table(rows, fields))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dual_gpu_launcher(path: Path, rows: list[dict[str, Any]]) -> None:
    train_rows = [row for row in rows if row["train_required"]]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"',
        'GPU_IDS="${GPU_IDS:-6,9}"',
        'NPROC_PER_NODE="${NPROC_PER_NODE:-2}"',
        'MASTER_PORT="${MASTER_PORT:-29693}"',
        'PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"',
        'PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"',
        'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"',
        'RESUME="${RESUME:-0}"',
        "",
        'if [[ "$GPU_IDS" != "6,9" ]]; then',
        '  echo "ERROR: FOCUS-EQText training mode requires physical GPUs 6,9; got GPU_IDS=$GPU_IDS" >&2',
        "  exit 2",
        "fi",
        'if [[ "$NPROC_PER_NODE" != "2" ]]; then',
        '  echo "ERROR: FOCUS-EQText training mode requires NPROC_PER_NODE=2; got $NPROC_PER_NODE" >&2',
        "  exit 2",
        "fi",
        "",
        'if [[ "$#" -eq 0 ]]; then',
        '  set -- "EQ_V30_dual_eqtext"',
        "fi",
        "",
        'cd "$ROOT_DIR"',
        "",
        "run_variant() {",
        '  local variant_id="$1"',
        '  local config_path="$2"',
        '  local work_dir="$3"',
        '  local resume_args=()',
        '  if [[ "$RESUME" == "1" ]]; then',
        '    resume_args+=(--resume)',
        '  fi',
        '  echo "[focus-eqtext] train $variant_id on physical GPUs $GPU_IDS resume=$RESUME"',
        '  env CUDA_VISIBLE_DEVICES="$GPU_IDS" \\',
        '    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \\',
        '    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \\',
        '    PORT="$MASTER_PORT" PYTHON="$PYTHON_BIN" \\',
        '    bash tools/my_dist_train.sh "$config_path" "$NPROC_PER_NODE" \\',
        '    --work-dir "$work_dir" "${resume_args[@]}"',
        "}",
        "",
        "for variant_id in \"$@\"; do",
        '  case "$variant_id" in',
    ]
    for row in train_rows:
        variant_id = row["variant_id"]
        config_path = row["config_py"]
        work_dir = str(Path("work_dirs") / "focus_eqtext_dota_short_20260609" / variant_id)
        lines.extend([
            f'    "{variant_id}")',
            f'      run_variant "{variant_id}" "{config_path}" "{work_dir}"',
            "      ;;",
        ])
    lines.extend([
        "    *)",
        '      echo "ERROR: unknown FOCUS-EQText variant: $variant_id" >&2',
        "      exit 2",
        "      ;;",
        "  esac",
        "done",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "configs")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    ensure_exp_tree(exp_dir)

    variants = variant_defs(resolve(repo_root, SPATIAL_TARGETS))
    rows: list[dict[str, Any]] = []
    for variant_id, variant in variants.items():
        json_path = output_dir / f"{variant_id}.json"
        py_path = output_dir / f"{variant_id}.py"
        payload = dict(variant)
        payload["config_py"] = str(py_path)
        payload["config_json"] = str(json_path)
        payload["base_config"] = str(resolve(repo_root, BASE_CONFIG))
        payload["max_iters"] = 1000
        write_json(json_path, payload)
        write_py_config(py_path, payload, exp_dir)
        focus_ovd = payload.get("focus_ovd", {})
        rows.append({
            "variant_id": variant_id,
            "support_type": payload["support_type"],
            "profile": payload["profile"],
            "use_focus_ovd": payload["use_focus_ovd"],
            "eqtext_enabled": bool(focus_ovd.get("eqtext", {}).get("enable", False)),
            "dual_enabled": bool(focus_ovd.get("dual_fusion", {}).get("enable", False)),
            "train_required": variant_id in TRAIN_VARIANTS,
            "target_mapping_mode": payload.get("focus_losses", {}).get(
                "target_mapping_mode", "none"),
            "config_json": str(json_path),
            "config_py": str(py_path),
        })

    write_csv(output_dir / "focus_eqtext_config_index.csv", rows)
    write_markdown(output_dir / "focus_eqtext_config_index.md", rows)
    write_dual_gpu_launcher(
        output_dir / "run_focus_eqtext_p2_train_gpu69.sh",
        rows)
    print(json.dumps({
        "status": "PASS_FOCUS_EQTEXT_CONFIGS_GENERATED",
        "variants": [row["variant_id"] for row in rows],
        "config_index": str(output_dir / "focus_eqtext_config_index.csv"),
        "dual_gpu_launcher": str(
            output_dir / "run_focus_eqtext_p2_train_gpu69.sh"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run 10 safe text-side 2ep variants and summarize AP."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PIPE124_PATH = Path(__file__).with_name(
    "124_text_fourier_detector_2ep_pipeline.py")
SPEC = importlib.util.spec_from_file_location("text_fourier_pipe124", PIPE124_PATH)
pipe124 = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(pipe124)


BASE_CONFIG = pipe124.BASE_CONFIG
BASELINE_CKPT = pipe124.BASELINE_CKPT
EXP_DIR = Path("resultmd/exp_text_side_safe_2ep_20260610")
DOTA2_CLASSES = pipe124.DOTA2_CLASSES
MONITORED_CLASSES = pipe124.MONITORED_CLASSES
KEY_CLASSES = pipe124.KEY_CLASSES
BASELINE_ROW = pipe124.BASELINE_ROW


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def tac_cfg(enable=False, use_alpha=True, use_beta=False,
            alpha_bound=0.02, beta_bound=0.05, classes=None,
            anchor_weight=0.01) -> dict[str, Any]:
    return {
        "num_classes": len(DOTA2_CLASSES),
        "class_names": DOTA2_CLASSES,
        "enable": bool(enable),
        "use_alpha": bool(use_alpha),
        "use_beta": bool(use_beta),
        "alpha_init": 0.0,
        "beta_init": 0.0,
        "alpha_bound": float(alpha_bound),
        "beta_bound": float(beta_bound),
        "apply_to_classes": list(classes or MONITORED_CLASSES),
        "anchor_weight": float(anchor_weight),
        "log_debug": True,
    }


def text_mixer_cfg(enable=False, mode="smooth", classes=None,
                   gamma_bound=0.02, beta_bound=0.04, use_beta=False,
                   sim_threshold=0.0, sim_power=1.0, topk=0,
                   anchor_weight=0.01) -> dict[str, Any]:
    return {
        "num_classes": len(DOTA2_CLASSES),
        "class_names": DOTA2_CLASSES,
        "enable": bool(enable),
        "mode": str(mode),
        "use_gamma": True,
        "use_beta": bool(use_beta),
        "gamma_init": 0.0,
        "beta_init": 0.0,
        "gamma_bound": float(gamma_bound),
        "beta_bound": float(beta_bound),
        "apply_to_classes": list(classes or MONITORED_CLASSES),
        "sim_threshold": float(sim_threshold),
        "sim_power": float(sim_power),
        "topk": int(topk),
        "anchor_weight": float(anchor_weight),
    }


def orientation_cfg() -> dict[str, Any]:
    return pipe124.orientation_cfg((2, 4, 6), min_confidence=0.20)


def base_focus() -> dict[str, Any]:
    return pipe124.base_focus(
        orientation_cfg(),
        pipe124.visual_adapter_cfg(False, ("small-vehicle",)))


def focus_losses() -> dict[str, Any]:
    return pipe124.focus_losses(False)


def head_gate_cfg() -> dict[str, Any]:
    return pipe124.head_gate_cfg(False)


def variant_defs() -> list[dict[str, Any]]:
    key = tuple(KEY_CLASSES)
    monitored = tuple(MONITORED_CLASSES)
    sv_lv_ship = ("small-vehicle", "large-vehicle", "ship")
    risk = ("small-vehicle", "bridge", "helipad", "ship")
    return [
        {
            "variant_id": "TSAFE_V01_TAC_ALPHA_KEY",
            "family": "text_calibration",
            "injection": "class_logit_alpha",
            "method": "zero-init text anchor alpha on key classes",
            "tac": tac_cfg(True, True, False, 0.015, classes=key),
            "text_mixer": text_mixer_cfg(False),
            "trainable": ["bbox_head.focus_text_anchor_calibration"],
        },
        {
            "variant_id": "TSAFE_V02_TAC_BETA_KEY",
            "family": "text_calibration",
            "injection": "class_logit_beta",
            "method": "zero-init text anchor beta on key classes",
            "tac": tac_cfg(True, False, True, beta_bound=0.030, classes=key),
            "text_mixer": text_mixer_cfg(False),
            "trainable": ["bbox_head.focus_text_anchor_calibration"],
        },
        {
            "variant_id": "TSAFE_V03_TAC_ALPHA_BETA_MON",
            "family": "text_calibration",
            "injection": "class_logit_alpha_beta",
            "method": "text anchor alpha plus beta on monitored classes",
            "tac": tac_cfg(True, True, True, 0.012, 0.025, monitored),
            "text_mixer": text_mixer_cfg(False),
            "trainable": ["bbox_head.focus_text_anchor_calibration"],
        },
        {
            "variant_id": "TSAFE_V04_MIX_SMOOTH_KEY",
            "family": "text_proto_mixer",
            "injection": "prototype_similarity_smooth",
            "method": "text prototype similarity smoothing on key classes",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "smooth", key, gamma_bound=0.012, topk=3),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V05_MIX_SUPPRESS_RISK",
            "family": "text_proto_mixer",
            "injection": "prototype_similarity_suppress",
            "method": "suppress similar text-prototype leakage on risk classes",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "suppress", risk, gamma_bound=0.012,
                sim_threshold=0.05, topk=4),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V06_MIX_SHARPEN_KEY",
            "family": "text_proto_mixer",
            "injection": "prototype_similarity_sharpen",
            "method": "text-similarity high-pass sharpening on key classes",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "sharpen", key, gamma_bound=0.010,
                sim_threshold=0.03, topk=4),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V07_MIX_SMOOTH_SV_LV_SHIP",
            "family": "text_proto_mixer",
            "injection": "prototype_similarity_smooth",
            "method": "targeted vehicle/ship text smoothing with top2 neighbors",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "smooth", sv_lv_ship, gamma_bound=0.010, topk=2),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V08_MIX_NORM_KEY",
            "family": "text_proto_mixer",
            "injection": "prototype_norm_scale",
            "method": "prototype norm based tiny text logit scaling on key classes",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "norm_scale", key, gamma_bound=0.010),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V09_MIX_CENTER_MON",
            "family": "text_proto_mixer",
            "injection": "class_centering",
            "method": "monitored-class mean-centering residual",
            "tac": tac_cfg(False),
            "text_mixer": text_mixer_cfg(
                True, "center", monitored, gamma_bound=0.008),
            "trainable": ["bbox_head.focus_text_logit_mixer"],
        },
        {
            "variant_id": "TSAFE_V10_TAC_MIX_COMBO",
            "family": "text_combo",
            "injection": "text_anchor_plus_proto_sharpen",
            "method": "small TAC beta with text-prototype sharpening",
            "tac": tac_cfg(True, False, True, beta_bound=0.020, classes=risk),
            "text_mixer": text_mixer_cfg(
                True, "sharpen", risk, gamma_bound=0.008,
                sim_threshold=0.03, topk=4),
            "trainable": [
                "bbox_head.focus_text_anchor_calibration",
                "bbox_head.focus_text_logit_mixer",
            ],
        },
    ]


def hook_lines(trainable: Sequence[str], exp_dir: Path,
               variant_id: str) -> list[str]:
    return pipe124.hook_lines(trainable, exp_dir, variant_id)


def write_config(path: Path, variant: dict[str, Any],
                 base_config: Path, checkpoint: Path,
                 exp_dir: Path) -> None:
    variant_id = variant["variant_id"]
    work_dir = exp_dir / "train" / variant_id / "work_dir"
    trainable = list(variant["trainable"])
    batch_size = int(variant.get("batch_size", 1))
    lines = [
        f"_base_ = {str(base_config)!r}",
        "",
        f"variant_id = {variant_id!r}",
        f"load_from = {str(checkpoint)!r}",
        f"work_dir = {str(work_dir)!r}",
        "num_gpus = 2",
        f"batch_size = {batch_size}",
        "max_iter_per_epoch = 400",
        "max_epochs = 2",
        "val_interval = 999",
        "source_prob = [1]",
        f"trainable_parameters = {repr(trainable)}",
        "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)",
        "param_scheduler = []",
        "optim_wrapper = dict(",
        "    type='OptimWrapper',",
        "    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0))",
        "default_hooks = dict(",
        "    logger=dict(type='LoggerHook', interval=50),",
        "    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=1),",
        ")",
        "",
        "model = dict(",
        "    support_type='text',",
        "    use_declip_support=False,",
        "    with_aux_bbox_head=True,",
        "    with_image_rec_losses=False,",
        "    bbox_head=dict(",
        "        use_focus_ovd=True,",
        f"        focus_ovd={repr(base_focus())},",
        f"        focus_losses={repr(focus_losses())},",
        f"        focus_text_anchor_calibration={repr(variant['tac'])},",
        f"        focus_fourier_head_gate={repr(head_gate_cfg())},",
        f"        focus_text_logit_mixer={repr(variant['text_mixer'])},",
        "    ),",
        ")",
        "",
        "train_dataloader = dict(",
        "    batch_size=batch_size,",
        "    sampler=dict(",
        "        batch_size=batch_size,",
        "        num_gpus=num_gpus,",
        "        max_iter_per_epoch=max_iter_per_epoch),",
        ")",
        "",
    ]
    lines.extend(hook_lines(trainable, exp_dir, variant_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_configs(repo_root: Path, exp_dir: Path,
                     base_config: Path, checkpoint: Path) -> list[dict[str, Any]]:
    pipe124.ensure_tree(exp_dir)
    rows = []
    for variant in variant_defs():
        config_py = exp_dir / "configs" / f"{variant['variant_id']}.py"
        config_json = exp_dir / "configs" / f"{variant['variant_id']}.json"
        write_config(config_py, variant, base_config, checkpoint, exp_dir)
        payload = dict(variant)
        payload.update({
            "config_py": str(config_py),
            "config_json": str(config_json),
            "base_config": str(base_config),
            "checkpoint": str(checkpoint),
            "support_type": "text",
            "with_aux_bbox_head": True,
            "use_declip_support": False,
            "max_epochs": 2,
            "batch_size": int(variant.get("batch_size", 1)),
        })
        write_json(config_json, payload)
        rows.append({
            "variant_id": variant["variant_id"],
            "family": variant["family"],
            "injection": variant["injection"],
            "method": variant["method"],
            "train_required": "true",
            "trainable_substrings": ",".join(variant["trainable"]),
            "support_type": "text",
            "with_aux_bbox_head": "true",
            "batch_size": str(int(variant.get("batch_size", 1))),
            "config_py": str(config_py),
            "config_json": str(config_json),
        })
    for name in (
            "text_side_safe_config_index.csv",
            "text_fourier_detector_config_index.csv"):
        write_csv(exp_dir / "configs" / name, rows)
    write_json(exp_dir / "configs" / "text_side_safe_config_index.json", {
        "status": "PASS_TEXT_SIDE_SAFE_CONFIGS",
        "variant_count": len(rows),
        "baseline": BASELINE_ROW,
        "variants": rows,
    })
    return rows


def fmt(value: Any) -> str:
    return pipe124.fmt(value)


def markdown_table(rows: Sequence[Mapping[str, Any]],
                   fields: Sequence[str]) -> list[str]:
    return pipe124.markdown_table(rows, fields)


def build_report(exp_dir: Path, repo_root: Path) -> Path:
    eval_rows = read_csv(
        exp_dir / "eval_epoch2" / "text_fourier_detector_eval_summary.csv")
    train_rows = read_csv(
        exp_dir / "train" / "text_fourier_detector_train_summary.csv")
    baseline = dict(BASELINE_ROW)
    baseline.update(pipe124.checkpoint_tensor_stats(
        pipe124.resolve(repo_root, BASELINE_CKPT)))
    baseline["param_delta_M"] = 0.0
    baseline["eval_status"] = "ZERO_REFERENCE"
    train_by_variant = {row.get("variant_id"): row for row in train_rows}
    all_rows: list[dict[str, Any]] = [baseline]
    for row in eval_rows:
        converted: dict[str, Any] = dict(row)
        for key, value in list(converted.items()):
            try:
                if value != "":
                    converted[key] = float(value)
            except (TypeError, ValueError):
                pass
        checkpoint = converted.get("checkpoint", "")
        if checkpoint:
            stats = pipe124.checkpoint_tensor_stats(
                pipe124.resolve(repo_root, checkpoint))
            converted.update(stats)
            if isinstance(converted.get("params_M"), float):
                converted["param_delta_M"] = round(
                    float(converted["params_M"])
                    - float(baseline["params_M"]), 4)
        train_item = train_by_variant.get(
            str(converted.get("variant_id", "")), {})
        converted["train_status"] = train_item.get("status", "")
        converted["train_note"] = train_item.get("note", "")
        all_rows.append(converted)

    pass_rows = [
        row for row in all_rows
        if row.get("eval_status") in {"ZERO_REFERENCE", "PASS"}
        and isinstance(row.get("mAP"), (int, float))
    ]
    best_variant = max(
        pass_rows[1:],
        key=lambda row: float(row["mAP"]),
        default=None)
    compact_rows = []
    for row in all_rows:
        compact_rows.append({
            "模型": "zero baseline"
            if row.get("variant_id") == "zero_baseline"
            else row.get("variant_id", ""),
            "参数量(M)": row.get("params_M", ""),
            "mAP": row.get("mAP", ""),
            "AP50": row.get("AP50", ""),
            "small_vehicle AP": row.get("small_vehicle_AP", ""),
            "large_vehicle AP": row.get("large_vehicle_AP", ""),
            "ship AP": row.get("ship_AP", ""),
            "bridge AP": row.get("bridge_AP", ""),
            "helipad AP": row.get("helipad_AP", ""),
            "ΔmAP": row.get("mAP_delta", ""),
            "状态": row.get("eval_status", ""),
        })
    lines = [
        "# Text-Side Safe 2ep Fast Validation",
        "",
        "Baseline is `epoch_24_weights_only.pth` zero reference with mAP 0.6957.",
        "",
        "All variants keep `support_type='text'`, `use_declip_support=False`, and `with_aux_bbox_head=True`. They do not switch to visual support and do not create dense per-anchor text support tensors.",
        "",
        "Parameter columns count checkpoint `state_dict` tensors. The `-0.6387M` delta versus the zero baseline comes from `with_image_rec_losses=False`, which drops the saved `rec_neck.*` tensors; the aux detection branch remains present and frozen.",
        "",
        "Training uses two-card jobs on `0,1`, `6,7`, and `8,9`; validation uses six cards `0,1,6,7,8,9`.",
        "",
        "## 结论",
        "",
        f"- 完成验证的 text-side 方法数：{len(pass_rows) - 1} / {len(all_rows) - 1}。",
        f"- 最好变体：{best_variant.get('variant_id') if best_variant else ''}，mAP={fmt(best_variant.get('mAP')) if best_variant else ''}，ΔmAP={fmt(best_variant.get('mAP_delta')) if best_variant else ''}。",
        "- 本轮方法都是 zero-init 或 identity-init 小改动，用于判断 text 端是否存在可利用的稳定微调空间。",
        "",
        "## 截图式 AP 大表",
        "",
    ]
    lines.extend(markdown_table(compact_rows, [
        "模型",
        "参数量(M)",
        "mAP",
        "AP50",
        "small_vehicle AP",
        "large_vehicle AP",
        "ship AP",
        "bridge AP",
        "helipad AP",
        "ΔmAP",
        "状态",
    ]))
    lines.extend([
        "",
        "## Full AP Table",
        "",
    ])
    lines.extend(markdown_table(all_rows, [
        "variant_id",
        "train_status",
        "eval_status",
        "family",
        "injection",
        "params_M",
        "param_delta_M",
        "state_keys",
        "mAP",
        "AP50",
        "small_vehicle_AP",
        "large_vehicle_AP",
        "ship_AP",
        "bridge_AP",
        "helipad_AP",
        "mAP_delta",
        "total_dets",
    ]))
    lines.extend([
        "",
        "## Method Table",
        "",
    ])
    lines.extend(markdown_table(all_rows[1:], [
        "variant_id",
        "family",
        "injection",
        "method",
        "eval_status",
        "checkpoint",
    ]))
    lines.extend([
        "",
        "## Train Status",
        "",
    ])
    lines.extend(markdown_table(train_rows, [
        "variant_id", "status", "gpu", "checkpoint", "train_log",
    ]))
    report_path = exp_dir / "reports" / "text_side_safe_2ep_final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fres_path = exp_dir / "fres_text_side_safe_2ep_final.md"
    fres_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=BASELINE_CKPT)
    parser.add_argument(
        "--stage",
        choices=("generate", "train", "eval", "report", "all"),
        default="all")
    parser.add_argument("--train-gpu-pairs", default="0,1;6,7;8,9")
    parser.add_argument("--eval-gpu", default="0,1,6,7,8,9")
    parser.add_argument("--base-port", type=int, default=30080)
    parser.add_argument("--eval-port", type=int, default=30180)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--no-run-missing", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    exp_dir = pipe124.resolve(repo_root, args.exp_dir)
    base_config = pipe124.resolve(repo_root, args.base_config)
    checkpoint = pipe124.resolve(repo_root, args.checkpoint)
    pipe124.ensure_tree(exp_dir)

    index_path = exp_dir / "configs" / "text_side_safe_config_index.csv"
    if args.stage in {"generate", "all"} or not index_path.exists():
        rows = generate_configs(repo_root, exp_dir, base_config, checkpoint)
    else:
        rows = read_csv(index_path)
    if args.only.strip():
        only = {item.strip() for item in args.only.split(",") if item.strip()}
        rows = [row for row in rows if row.get("variant_id") in only]
        missing = sorted(only - {row.get("variant_id") for row in rows})
        if missing:
            raise ValueError(f"unknown --only variant ids: {missing}")

    if args.stage in {"train", "all"}:
        pipe124.train_variants(
            repo_root,
            exp_dir,
            rows,
            pipe124.parse_gpu_pairs(args.train_gpu_pairs),
            args.base_port)
    if args.stage in {"eval", "all"}:
        pipe124.eval_variants(
            repo_root,
            exp_dir,
            rows,
            args.eval_gpu,
            args.eval_port,
            args.seed,
            force=args.force_eval,
            run_missing=not args.no_run_missing)
    if args.stage in {"report", "all"}:
        report_path = build_report(exp_dir, repo_root)
    else:
        report_path = None
    print(json.dumps({
        "status": "PASS_TEXT_SIDE_SAFE_PIPELINE_STAGE",
        "stage": args.stage,
        "exp_dir": str(exp_dir),
        "variant_count": len(rows),
        "report": str(report_path or ""),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

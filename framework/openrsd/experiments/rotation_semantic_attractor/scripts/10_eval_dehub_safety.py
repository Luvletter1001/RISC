#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import traceback
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, add_common_args
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from experiments.rotation_semantic_attractor.src.metrics.dehub_safety import (
    class_distribution_from_histograms,
    js_kl_divergence,
    mean_float,
)
from experiments.rotation_semantic_attractor.src.metrics.proxy_experiments import dehub_safety_rows
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_adapter import OpenRSDAdapter
from experiments.rotation_semantic_attractor.src.utils.io import read_csv, write_csv
from experiments.rotation_semantic_attractor.src.utils.status import CLAIM_NONE, ExperimentStatus, status_metadata


BASE_CONFIG = PROJECT_ROOT / "M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py"
BASE_CHECKPOINT = PROJECT_ROOT / "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth"
REPAIR_CONFIG = PROJECT_ROOT / "M_configs/experiments/sv_dehub_lite_v1.py"
REPAIR_CANDIDATES = [
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_3000.pth",
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_2000.pth",
    PROJECT_ROOT / "work_dirs/exp_sv_dehub_lite_train_gpu89/train_v1/iter_1000.pth",
    PROJECT_ROOT / "work_dirs/exp_next_plan_sv_dehub_step23_overnight/branch_B_fresh_epoch24_to8k/iter_8000.pth",
]
SUPPORT_PKL = PROJECT_ROOT / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"


def _default_repair_checkpoint() -> Path:
    for path in REPAIR_CANDIDATES:
        if path.exists():
            return path
    return REPAIR_CANDIDATES[0]


def _selected_metric_rows(run_dir: Path, limit: int) -> list[dict]:
    path = run_dir / "metrics" / "false_hub_tile_angle.csv"
    rows = [row for row in read_csv(path) if row.get("region_mode") == "all_region"] if path.exists() else []
    if not limit or len(rows) <= limit:
        return rows
    high = [row for row in rows if int(float(row.get("num_gt_sv") or 0)) > 0]
    low = [row for row in rows if int(float(row.get("num_gt_sv") or 0)) == 0]
    selected = []
    if high:
        selected.append(high[0])
    if low and len(selected) < limit:
        selected.append(low[0])
    for row in rows:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected


def _write_openrsd_ann_pkl(txt_path: Path, pkl_path: Path) -> int:
    records = [record for record in read_dota_txt(txt_path) if "polygon" in record]
    texts = [record["class_name"] for record in records]
    polys = []
    for record in records:
        flat = []
        for point in record["polygon"]:
            flat.extend([float(point[0]), float(point[1])])
        polys.append(flat)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "polys": np.asarray(polys, dtype=np.float32),
                "text_embeds": None,
                "visual_embeds": None,
                "cls_list": texts,
                "ict_support_dict": None,
            },
            f,
        )
    return len(records)


def _prepare_openrsd_eval_dataset(run_dir: Path, rows: list[dict]) -> dict:
    dataset_root = run_dir / "dehub_safety_openrsd_dataset"
    images_dir = dataset_root / "images"
    ann_dir = dataset_root / "annfiles"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        model = row["model_name"]
        tile_id = row["tile_id"]
        angle = int(row["angle"])
        raw_path = run_dir / "raw_predictions" / model / tile_id / f"angle_{angle:03d}.json"
        if not raw_path.exists():
            continue
        raw = json.loads(raw_path.read_text())
        image_path = Path(raw.get("metadata", {}).get("rotated_image_path", ""))
        ann_path = Path(raw.get("metadata", {}).get("rotated_gt_path", ""))
        if not image_path.exists() or not ann_path.exists():
            continue
        stem = f"{tile_id}_rot{angle:03d}"
        new_image = images_dir / f"{stem}.png"
        new_ann = ann_dir / f"{stem}.pkl"
        shutil.copy2(image_path, new_image)
        num_instances = _write_openrsd_ann_pkl(ann_path, new_ann)
        items.append(
            {
                "stem": stem,
                "tile_id": tile_id,
                "angle": angle,
                "group": "high" if int(float(row.get("num_gt_sv") or 0)) > 0 else "low",
                "num_gt_sv": int(float(row.get("num_gt_sv") or 0)),
                "num_gt_total": int(float(row.get("num_gt_total") or 0)),
                "image_path": str(new_image),
                "ann_path": str(new_ann),
                "num_instances": num_instances,
            }
        )
    return {"dataset_root": str(dataset_root), "images_dir": str(images_dir), "ann_dir": str(ann_dir), "items": items}


def _eval_openrsd_checkpoint(name: str, config: Path, checkpoint: Path, dataset: dict, out_dir: Path, device: str) -> list[dict]:
    import torch
    from tools.rotation_diagnostics import probe_rotated_stage_outputs as base
    from tools.rotation_overnight_gpu89 import common as C

    angles = sorted({item["angle"] for item in dataset["items"]})
    item_by_stem = {item["stem"]: item for item in dataset["items"]}
    cfg = {
        "name": name,
        "model_type": "openrsd",
        "box_type": "obb",
        "adapter": "openrsd",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "support_pkl": str(SUPPORT_PKL),
        "support_type": "visual",
        "support_shot": 8,
        "score_thr": 0.3,
        "iou_thr": 0.5,
        "result_md_dir": str(out_dir / "reports"),
    }
    adapter = OpenRSDAdapter(name, cfg, PROJECT_ROOT)
    adapter.load(device=device, image_dir=dataset["images_dir"], out_dir=out_dir / name, angles=angles)
    loader = adapter.build_loader(dataset["images_dir"], angles)
    runtime = adapter.runtime
    assert runtime is not None
    rows = []
    with torch.no_grad():
        for data_info in loader:
            img_path = data_info["data_samples"][0].img_path
            angle = base.angle_from_name(img_path)
            stem = Path(img_path).stem
            item = item_by_stem.get(stem, {})
            data = runtime.model.data_preprocessor(data_info, False)
            data["inputs"] = data["inputs"].to(runtime.device)
            samples = data["data_samples"]
            features = runtime.model.prompt_extract_feats(data["inputs"])
            metas = [sample.metainfo for sample in samples]
            support_feats, support_labels, _ = C.build_support(
                runtime.model,
                runtime.base_args,
                runtime.support_data,
                runtime.name2id,
                runtime.device,
                "original",
            )
            outs = C.model_forward_dense(runtime.model, features, support_feats, support_labels, runtime.base_args)
            summary = C.post_summary(angle, name, runtime.model.bbox_head, outs, metas)
            rows.append(
                {
                    **item,
                    "checkpoint_name": name,
                    "checkpoint_path": str(checkpoint),
                    "detection_total": summary["detection_total"],
                    "small_vehicle_count": summary["small_vehicle_count"],
                    "large_vehicle_count": summary["large_vehicle_count"],
                    "small_vehicle_ratio": summary["small_vehicle_ratio"],
                    "large_vehicle_ratio": summary["large_vehicle_ratio"],
                    "mean_score": summary["mean_score"],
                    "max_score": summary["max_score"],
                    "top1_class": summary["top1_class"],
                    "class_histogram": summary["class_histogram"],
                }
            )
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _real_dehub_rows(run_dir: Path, args) -> list[dict]:
    selected = _selected_metric_rows(run_dir, args.limit or 2)
    dataset = _prepare_openrsd_eval_dataset(run_dir, selected)
    if not dataset["items"]:
        meta = status_metadata(ExperimentStatus.NOT_AVAILABLE_ASSET, "no smoke images available for DeHub safety inference", claim_level=CLAIM_NONE)
        return [{"method": "DeHub safety smoke", "actual_inference_run": False, "is_schema_only": False, **meta}]

    baseline_config = Path(args.baseline_config)
    baseline_checkpoint = Path(args.baseline_checkpoint)
    repair_config = Path(args.repair_config)
    repair_checkpoint = Path(args.repair_checkpoint)
    out_dir = run_dir / "dehub_safety_eval"
    baseline_rows = _eval_openrsd_checkpoint("baseline", baseline_config, baseline_checkpoint, dataset, out_dir, args.device)
    repair_rows = _eval_openrsd_checkpoint("repair", repair_config, repair_checkpoint, dataset, out_dir, args.device)
    write_csv(run_dir / "metrics" / "dehub_safety_baseline_raw.csv", baseline_rows)
    write_csv(run_dir / "metrics" / "dehub_safety_repair_raw.csv", repair_rows)

    base_dist = class_distribution_from_histograms(baseline_rows)
    repair_dist = class_distribution_from_histograms(repair_rows)
    class_js, class_kl = js_kl_divergence(base_dist, repair_dist)
    base_high = [row for row in baseline_rows if row.get("group") == "high"]
    repair_high = [row for row in repair_rows if row.get("group") == "high"]
    base_low = [row for row in baseline_rows if row.get("group") == "low"]
    repair_low = [row for row in repair_rows if row.get("group") == "low"]
    base_high_sv = mean_float(base_high, "small_vehicle_ratio")
    repair_high_sv = mean_float(repair_high, "small_vehicle_ratio")
    base_low_sv = mean_float(base_low, "small_vehicle_ratio")
    repair_low_sv = mean_float(repair_low, "small_vehicle_ratio")
    lowrisk_inflation = repair_low_sv - base_low_sv if base_low and repair_low else ""
    meta = status_metadata(
        ExperimentStatus.DONE_SMOKE,
        "baseline and DeHub repair checkpoints were both run on the same smoke split/angles; not a full safety benchmark",
        claim_level=CLAIM_NONE,
        is_scientific_result=False,
        include_in_main_table=False,
    )
    return [
        {
            "method": "OpenRSD DeHub safety smoke",
            "baseline_checkpoint": str(baseline_checkpoint),
            "repair_checkpoint": str(repair_checkpoint),
            "baseline_config": str(baseline_config),
            "repair_config": str(repair_config),
            "same_split": True,
            "same_angles": True,
            "same_threshold": True,
            "same_evaluator": True,
            "actual_inference_run": True,
            "has_false_hub_metrics": True,
            "has_true_sv_preservation": False,
            "has_det_per_img": True,
            "has_class_distribution_js_kl": True,
            "has_lowrisk_inflation": bool(base_low and repair_low),
            "has_ap50_or_proxy": False,
            "is_schema_only": False,
            "include_in_repair_table": False,
            "mAP50": "",
            "SV_AP50": "",
            "Dense_or_QAR_SV": "",
            "Final_FSV": repair_high_sv,
            "Baseline_HighRisk_SV": base_high_sv,
            "Repair_HighRisk_SV": repair_high_sv,
            "Baseline_LowRisk_SV": base_low_sv,
            "Repair_LowRisk_SV": repair_low_sv,
            "LowRiskInflation": lowrisk_inflation,
            "Class_JS": class_js,
            "Class_KL": class_kl,
            "baseline_raw_csv": str(run_dir / "metrics" / "dehub_safety_baseline_raw.csv"),
            "repair_raw_csv": str(run_dir / "metrics" / "dehub_safety_repair_raw.csv"),
            "evaluated_images": len(dataset["items"]),
            "Verdict": ExperimentStatus.DONE_SMOKE,
            **meta,
            "unsupported_reason": "",
        }
    ]


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["auto", "real", "schema"], default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-config", default=str(BASE_CONFIG))
    parser.add_argument("--baseline-checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--repair-config", default=str(REPAIR_CONFIG))
    parser.add_argument("--repair-checkpoint", default=str(_default_repair_checkpoint()))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics_dir = Path(args.output_dir or run_dir / "metrics")
    if args.mode in {"auto", "real"}:
        try:
            base_rows = _real_dehub_rows(run_dir, args)
            write_csv(metrics_dir / "dehub_safety.csv", base_rows)
            write_csv(metrics_dir / "class_drift.csv", base_rows)
            write_csv(metrics_dir / "lowrisk_inflation.csv", base_rows)
            write_csv(metrics_dir / "true_sv_preservation.csv", base_rows)
            print(f"dehub_safety={metrics_dir / 'dehub_safety.csv'}")
            return
        except Exception as exc:
            if args.mode == "real":
                raise
            meta = status_metadata(
                ExperimentStatus.FAILED,
                f"real DeHub safety evaluation failed: {type(exc).__name__}:{exc}",
                claim_level=CLAIM_NONE,
            )
            base_rows = [
                {
                    "method": "OpenRSD DeHub safety smoke",
                    "baseline_checkpoint": str(args.baseline_checkpoint),
                    "repair_checkpoint": str(args.repair_checkpoint),
                    "same_split": False,
                    "actual_inference_run": False,
                    "is_schema_only": False,
                    "include_in_repair_table": False,
                    **meta,
                    "unsupported_reason": meta["status_reason"],
                    "traceback": traceback.format_exc(),
                }
            ]
            write_csv(metrics_dir / "dehub_safety.csv", base_rows)
            write_csv(metrics_dir / "class_drift.csv", base_rows)
            write_csv(metrics_dir / "lowrisk_inflation.csv", base_rows)
            write_csv(metrics_dir / "true_sv_preservation.csv", base_rows)
            print(f"dehub_safety={metrics_dir / 'dehub_safety.csv'}")
            return

    false_hub_path = metrics_dir / "false_hub_summary_by_model.csv"
    stage_path = metrics_dir / "stage_decomposition_summary.csv"
    stage_by_model = {row["model_name"]: row for row in read_csv(stage_path)} if stage_path.exists() else {}
    base_rows = []
    if false_hub_path.exists():
        for row in read_csv(false_hub_path):
            model = row["model_name"]
            base_rows.extend(dehub_safety_rows(model, row, stage_by_model.get(model, {})))
    if not base_rows:
        meta = status_metadata(
            ExperimentStatus.NOT_AVAILABLE_ASSET,
            "requires both baseline and DeHub checkpoint",
            claim_level=CLAIM_NONE,
        )
        base_rows = [
            {
                "method": method,
                "baseline_checkpoint": "",
                "repair_checkpoint": "",
                "same_split": False,
                "actual_inference_run": False,
                "has_true_sv_preservation": False,
                "has_class_drift": False,
                "has_lowrisk_inflation": False,
                "is_schema_only": True,
                "include_in_repair_table": False,
                "mAP50": "",
                "SV_AP50": "",
                "Dense_or_QAR_SV": "",
                "Final_FSV": "",
                "ObjectFlip_SV": "",
                "BG_FSV": "",
                "True_SV_Recall": "",
                "Class_JS": "",
                "LowRiskInflation": "",
                "Verdict": ExperimentStatus.NOT_AVAILABLE_ASSET,
                **meta,
                "unsupported_reason": meta["status_reason"],
            }
            for method in ["OpenRSD Step2", "OpenRSD Step3", "DeHub B5k", "DeHub B8k", "rotation TTA"]
        ]
    write_csv(metrics_dir / "dehub_safety.csv", base_rows)
    write_csv(metrics_dir / "class_drift.csv", base_rows)
    write_csv(metrics_dir / "lowrisk_inflation.csv", base_rows)
    write_csv(metrics_dir / "true_sv_preservation.csv", base_rows)
    print(f"dehub_safety={metrics_dir / 'dehub_safety.csv'}")


if __name__ == "__main__":
    main()

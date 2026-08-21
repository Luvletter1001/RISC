#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import traceback
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, add_common_args, exp_path, parse_angles
from experiments.rotation_semantic_attractor.src.dota_io import read_dota_txt
from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_hooks import tensor_checksum
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_adapter import OpenRSDAdapter
from experiments.rotation_semantic_attractor.src.model_adapters.openrsd_hook_registry import OpenRSDHookRecorder
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv, write_json
from experiments.rotation_semantic_attractor.src.utils.status import (
    CLAIM_ENGINEERING_SMOKE,
    CLAIM_NONE,
    ExperimentStatus,
    status_metadata,
)


def _first_openrsd_pair(inventory_path: Path) -> dict | None:
    inventory = read_json(inventory_path)
    for pair in inventory.get("runnable_pairs", []):
        if pair.get("candidate_model_family") == "openrsd" and pair.get("usable", True):
            return pair
    return None


def _angle_from_smoke_name(path: Path) -> int | None:
    match = re.search(r"_angle_(\d{3})", path.stem)
    return int(match.group(1)) if match else None


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


def prepare_openrsd_dataset(source_run_dir: Path, out_dir: Path, angles: list[int], limit: int) -> dict:
    image_root = source_run_dir / "rotated_images"
    gt_root = source_run_dir / "rotated_gt"
    dataset_root = out_dir / "openrsd_dataset"
    images_dir = dataset_root / "images"
    ann_dir = dataset_root / "annfiles"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for image_path in sorted(image_root.glob("*/*_angle_*.png")):
        angle = _angle_from_smoke_name(image_path)
        if angle is None or angle not in angles:
            continue
        tile_id = image_path.parent.name
        gt_path = gt_root / tile_id / f"{image_path.stem}.txt"
        if not gt_path.exists():
            continue
        new_stem = f"{tile_id}_rot{angle:03d}"
        new_image = images_dir / f"{new_stem}.png"
        new_ann = ann_dir / f"{new_stem}.pkl"
        shutil.copy2(image_path, new_image)
        num_instances = _write_openrsd_ann_pkl(gt_path, new_ann)
        copied.append(
            {
                "tile_id": tile_id,
                "angle": angle,
                "image_path": str(new_image),
                "ann_path": str(new_ann),
                "num_instances": num_instances,
            }
        )
        if limit and len(copied) >= limit:
            break
    return {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "ann_dir": str(ann_dir),
        "items": copied,
    }


def _row_status(status: str, reason: str, claim: str = CLAIM_NONE) -> dict:
    return status_metadata(status, reason, claim_level=claim, is_scientific_result=False, include_in_main_table=False)


def _render_report(manifest: dict, det_rows: list[dict], hook_rows: list[dict]) -> str:
    hook_done = sum(1 for row in hook_rows if row.get("status") == ExperimentStatus.DONE_SMOKE)
    lines = [
        "# OpenRSD Hook Smoke",
        "",
        f"- status: `{manifest['status']}`",
        f"- status_reason: `{manifest['status_reason']}`",
        f"- config: `{manifest.get('config', '')}`",
        f"- checkpoint: `{manifest.get('checkpoint', '')}`",
        f"- support_pkl: `{manifest.get('support_pkl', '')}`",
        f"- copied_images: `{len(manifest.get('dataset', {}).get('items', []))}`",
        f"- detection_rows: `{len(det_rows)}`",
        f"- hook_rows: `{len(hook_rows)}`",
        f"- hook_rows_done_smoke: `{hook_done}`",
        "",
        "## DONE_FULL Gate",
        "",
        "- This smoke is not DONE_FULL: it uses a tiny split/angle subset and is only adapter/hook verification.",
        "- Embedding edits are feature-level visual support changes, not prompt-only text changes.",
        "",
        "## Hook Targets",
        "",
        "| target | module | status | shape |",
        "| --- | --- | --- | --- |",
    ]
    for row in hook_rows[:50]:
        lines.append(
            f"| {row.get('hook_target', '')} | `{row.get('module_name', '')}` | {row.get('status', '')} | `{row.get('tensor_shape_example', '')}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument(
        "--inventory",
        default=str(exp_path("outputs", "open_vocab_assets", "open_vocab_asset_inventory.json")),
    )
    parser.add_argument(
        "--source-run-dir",
        default=str(exp_path("outputs", "runs", "smoke_2tiles_2angles_gpu_hooks")),
    )
    parser.add_argument("--report-dir", default=str(exp_path("reports")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interventions", default="original,zero_sv")
    args = parser.parse_args()

    angles = parse_angles(args.angles, default=[0, 90])
    interventions = [item.strip() for item in args.interventions.split(",") if item.strip()]
    out_dir = Path(args.output_dir or exp_path("outputs", "open_vocab_assets", "openrsd_hook_smoke"))
    metrics_dir = out_dir / "metrics"
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    pair = _first_openrsd_pair(Path(args.inventory))
    manifest = {
        "command": "15_openrsd_hook_smoke.py",
        "args": vars(args),
        "angles": angles,
        "interventions": interventions,
        "status": ExperimentStatus.NOT_RUN,
        "status_reason": "",
        "failures": [],
    }
    if pair is None:
        meta = _row_status(
            ExperimentStatus.NOT_AVAILABLE_ASSET,
            "no runnable OpenRSD config/checkpoint/support pair in open-vocabulary inventory",
        )
        manifest.update(meta)
        write_json(out_dir / "manifest.json", manifest)
        print(f"status={meta['status']}")
        print(f"manifest={out_dir / 'manifest.json'}")
        return

    dataset = prepare_openrsd_dataset(Path(args.source_run_dir), out_dir, angles, args.limit or 2)
    cfg = {
        "name": "OpenRSD inventory smoke",
        "model_type": "openrsd",
        "box_type": "obb",
        "adapter": "openrsd",
        "config": pair["config"],
        "checkpoint": pair["checkpoint"],
        "support_pkl": pair["support_pkl"],
        "prompt_protocol": str(exp_path("configs", "open_vocab_prompt_protocol.json")),
        "support_type": "visual",
        "support_shot": 8,
        "score_thr": 0.3,
        "iou_thr": 0.5,
        "result_md_dir": str(report_dir),
    }
    manifest.update(
        {
            "config": cfg["config"],
            "checkpoint": cfg["checkpoint"],
            "support_pkl": cfg["support_pkl"],
            "prompt_protocol": cfg["prompt_protocol"],
            "dataset": dataset,
        }
    )
    if not dataset["items"]:
        meta = _row_status(ExperimentStatus.NOT_AVAILABLE_ASSET, "no rotated smoke images/annfiles available for OpenRSD hook smoke")
        manifest.update(meta)
        write_json(out_dir / "manifest.json", manifest)
        print(f"status={meta['status']}")
        print(f"manifest={out_dir / 'manifest.json'}")
        return

    det_rows: list[dict] = []
    norm_rows: list[dict] = []
    hook_rows: list[dict] = []
    try:
        import torch
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as base
        from tools.rotation_overnight_gpu89 import common as C

        adapter = OpenRSDAdapter("openrsd_inventory_smoke", cfg, PROJECT_ROOT)
        adapter.load(device=args.device, image_dir=dataset["images_dir"], out_dir=out_dir, angles=angles)
        loader = adapter.build_loader(dataset["images_dir"], angles)
        runtime = adapter.runtime
        assert runtime is not None

        recorder = OpenRSDHookRecorder(max_spatial_size=32)
        hook_plan = recorder.register(runtime.model)
        base_support, _, _ = C.build_support(
            runtime.model,
            runtime.base_args,
            runtime.support_data,
            runtime.name2id,
            runtime.device,
            "original",
        )
        base_checksum = tensor_checksum(base_support)
        processed = 0
        with torch.no_grad():
            for data_info in loader:
                if args.limit and processed >= args.limit:
                    break
                img_path = data_info["data_samples"][0].img_path
                angle = base.angle_from_name(img_path)
                if angle not in angles:
                    continue
                image_name = Path(img_path).stem
                data = runtime.model.data_preprocessor(data_info, False)
                data["inputs"] = data["inputs"].to(runtime.device)
                samples = data["data_samples"]
                features = runtime.model.prompt_extract_feats(data["inputs"])
                metas = [sample.metainfo for sample in samples]
                for intervention in interventions:
                    recorder.clear()
                    support_feats, support_labels, norm = C.build_support(
                        runtime.model,
                        runtime.base_args,
                        runtime.support_data,
                        runtime.name2id,
                        runtime.device,
                        intervention,
                    )
                    norm_rows.extend(norm)
                    checksum = tensor_checksum(support_feats)
                    outs = C.model_forward_dense(runtime.model, features, support_feats, support_labels, runtime.base_args)
                    row = C.post_summary(angle, intervention, runtime.model.bbox_head, outs, metas)
                    row.update(
                        {
                            "model_name": "openrsd_inventory_smoke",
                            "model_family": "openrsd",
                            "image": image_name,
                            "has_open_vocab_config": True,
                            "has_checkpoint": True,
                            "has_prompt_or_class_embedding_path": True,
                            "is_prompt_only": False,
                            "is_embedding_level": True,
                            "is_visual_support_level": True,
                            "actual_embedding_modified": checksum != base_checksum,
                            "original_embedding_checksum": base_checksum,
                            "modified_embedding_checksum": checksum,
                            "modified_tensor_name": "visual_support_embeddings",
                            "reran_inference": True,
                            "has_paired_comparison": intervention != "original",
                            "is_smoke_limit": True,
                            **_row_status(
                                ExperimentStatus.DONE_SMOKE,
                                "real OpenRSD forward smoke with hook capture; tiny split is not a full scientific benchmark",
                                CLAIM_ENGINEERING_SMOKE,
                            ),
                        }
                    )
                    det_rows.append(row)
                    for hook_row in recorder.to_rows():
                        hook_rows.append(
                            {
                                "image": image_name,
                                "angle": angle,
                                "intervention": intervention,
                                **hook_row,
                            }
                        )
                    print(f"DONE openrsd_hook_smoke image={image_name} angle={angle:03d} intervention={intervention}")
                processed += 1
        recorder.close()
        meta = _row_status(
            ExperimentStatus.DONE_SMOKE,
            "OpenRSD assets loaded, real forward inference ran, and hook tensors were captured on a smoke subset",
            CLAIM_ENGINEERING_SMOKE,
        )
        manifest.update(meta)
        manifest["hook_plan"] = hook_plan
        manifest["processed_images"] = processed
    except Exception as exc:
        meta = _row_status(ExperimentStatus.FAILED, f"{type(exc).__name__}:{exc}")
        manifest.update(meta)
        manifest["failures"].append({"reason": meta["status_reason"], "traceback": traceback.format_exc()})
    finally:
        write_json(out_dir / "manifest.json", manifest)
        write_json(out_dir / "hook_rows.json", hook_rows)
        write_csv(metrics_dir / "open_vocab_intervention.csv", det_rows)
        write_csv(metrics_dir / "openrsd_hook_rows.csv", hook_rows)
        write_csv(metrics_dir / "openrsd_embedding_norms_before_after.csv", norm_rows)
        report_path = report_dir / "openrsd_hook_smoke.md"
        report_path.write_text(_render_report(manifest, det_rows, hook_rows))

    print(f"status={manifest['status']}")
    print(f"manifest={out_dir / 'manifest.json'}")
    print(f"report={report_dir / 'openrsd_hook_smoke.md'}")


if __name__ == "__main__":
    main()

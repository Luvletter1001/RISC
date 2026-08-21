#!/usr/bin/env python3
"""Fresh P0148 discovery reproduction with real support interventions.

This diagnostic is intentionally separate from the scene-disjoint N0 gate.
It reuses the frozen OpenRSD model and the established support intervention
implementation; it does not implement or train RISC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def prioritize_model_code_root(path):
    code_root = str(Path(path).resolve())
    sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != code_root]
    sys.path.insert(0, code_root)


def is_selected_view(path, *, angles, filename_token):
    stem = Path(path).stem
    match = re.search(r"_rot(\d{3})", stem)
    return bool(
        match
        and int(match.group(1)) in angles
        and (not filename_token or filename_token in stem)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-code-root", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-pkl", required=True)
    parser.add_argument("--normalized-class-dict", required=True)
    parser.add_argument("--neg-support-data", required=True)
    parser.add_argument("--pca-meta", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--angles", nargs="+", type=int, required=True)
    parser.add_argument("--interventions", nargs="+", default=[
        "original", "zero_sv", "swap_sv_lv", "random_sv"
    ])
    parser.add_argument("--filename-token", default="_A_original")
    parser.add_argument("--support-shot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner
    import torch

    from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_hooks import (
        tensor_checksum,
    )
    from tools.rotation_diagnostics import probe_rotated_stage_outputs as base
    from tools.rotation_overnight_gpu89 import common as common

    if args.model_code_root:
        if not Path(args.model_code_root).is_dir():
            raise FileNotFoundError(
                f"model code root does not exist: {args.model_code_root}"
            )
        prioritize_model_code_root(args.model_code_root)

    required = [
        args.config,
        args.checkpoint,
        args.support_pkl,
        args.normalized_class_dict,
        args.neg_support_data,
        args.pca_meta,
        args.image_dir,
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing runtime assets: {missing}")

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    angles = sorted({angle % 360 for angle in args.angles})

    base_args = SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        out_dir=str(out_dir),
        angles=angles,
        angle_step=None,
        score_thr=0.3,
        iou_thr=0.5,
        support_shot=args.support_shot,
        support_type="visual",
        support_feat=args.support_pkl,
        normalized_class_dict=args.normalized_class_dict,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        device=args.device,
        max_spatial_size=32,
        result_md_dir=str(report_dir),
    )
    base.setup_reproducibility(args.seed)
    cfg = base.build_cfg(base_args, Path(args.support_pkl))
    cfg.work_dir = str(out_dir / "runner_work_dir")
    cfg.load_from = args.checkpoint
    cfg.model.support_feat_dict = {"Data1_DOTA1": args.support_pkl}
    cfg.model.normalized_class_dict = args.normalized_class_dict
    cfg.model.neg_support_data = args.neg_support_data
    cfg.model.pca_meta_pth = args.pca_meta
    cfg.model.val_using_aux = False
    runner = Runner.from_cfg(cfg) if "runner_type" not in cfg else RUNNERS.build(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    model = runner.model
    device = torch.device(args.device)
    model.to(device).eval()
    _, _, support_data, name2id, _ = base.prepare_support(base_args, device)
    loader = base.build_dataloader(base_args, angles)

    detection_rows = []
    dense_rows = []
    norm_rows = []
    selected_images = []
    original_boxes = {}
    original_checksum = None
    with torch.no_grad():
        for data_info in loader:
            img_path = data_info["data_samples"][0].img_path
            if not is_selected_view(
                img_path,
                angles=set(angles),
                filename_token=args.filename_token,
            ):
                continue
            angle = base.angle_from_name(img_path)
            image_name = Path(img_path).stem
            selected_images.append(image_name)
            data = model.data_preprocessor(data_info, False)
            data["inputs"] = data["inputs"].to(device)
            samples = data["data_samples"]
            features = model.prompt_extract_feats(data["inputs"])
            metas = [sample.metainfo for sample in samples]
            for intervention in args.interventions:
                support_feats, support_labels, norms = common.build_support(
                    model,
                    base_args,
                    support_data,
                    name2id,
                    device,
                    intervention,
                )
                checksum = tensor_checksum(support_feats)
                if intervention == "original":
                    original_checksum = checksum
                for row in norms:
                    norm_rows.append({"angle": angle, **row})
                outs = common.model_forward_dense(
                    model, features, support_feats, support_labels, base_args
                )
                det_row = common.post_summary(
                    angle, intervention, model.bbox_head, outs, metas
                )
                det_row.update(
                    image=image_name,
                    support_checksum=checksum,
                    embedding_modified=(
                        False if original_checksum is None else checksum != original_checksum
                    ),
                )
                detection_rows.append(det_row)

                levels = common.decode_dense(
                    model.bbox_head, outs[0], outs[1], outs[2], metas[0]["img_shape"]
                )
                scores = torch.cat([level[2] for level in levels])
                boxes = torch.cat([level[3] for level in levels])
                top2 = scores.topk(2, dim=1).values
                if intervention == "original":
                    original_boxes[angle] = boxes.clone()
                reference_boxes = original_boxes.get(angle)
                if reference_boxes is None or reference_boxes.shape != boxes.shape:
                    mean_box_delta = float("nan")
                    max_box_delta = float("nan")
                else:
                    box_delta = (boxes - reference_boxes).abs()
                    mean_box_delta = float(box_delta.mean().item())
                    max_box_delta = float(box_delta.max().item())
                dense_rows.append(
                    dict(
                        image=image_name,
                        angle=angle,
                        intervention=intervention,
                        num_locations=int(scores.shape[0]),
                        dense_top1_small_vehicle_ratio=float(
                            (scores.argmax(dim=1) == common.SMALL).float().mean().item()
                        ),
                        small_vehicle_score_mean=float(
                            scores[:, common.SMALL].mean().item()
                        ),
                        mean_top1_margin=float((top2[:, 0] - top2[:, 1]).mean().item()),
                        decoded_box_mean_abs_delta_vs_original=mean_box_delta,
                        decoded_box_max_abs_delta_vs_original=max_box_delta,
                    )
                )
                print(
                    f"DONE P0148 angle={angle:03d} intervention={intervention} "
                    f"dense_sv={dense_rows[-1]['dense_top1_small_vehicle_ratio']:.6f} "
                    f"post_sv={det_row['small_vehicle_ratio']:.6f}"
                )

    expected = len(angles)
    if len(selected_images) != expected:
        raise RuntimeError(
            f"expected {expected} selected views, got {len(selected_images)}: {selected_images}"
        )
    if not detection_rows or not dense_rows:
        raise RuntimeError("no P0148 inference rows were produced")

    def write_csv(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "ftable_detection_summary.csv", detection_rows)
    write_csv(out_dir / "ftable_dense_semantic_summary.csv", dense_rows)
    write_csv(out_dir / "ftable_embedding_norms.csv", norm_rows)

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "status": "DONE_DISCOVERY",
        "scope": "P0148 discovery-only; excluded from N0-RI thresholds",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "model_code_root": args.model_code_root,
        "angles": angles,
        "interventions": args.interventions,
        "selected_images": selected_images,
        "assets": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "support_pkl": args.support_pkl,
            "support_sha256": sha256(args.support_pkl),
            "normalized_class_dict": args.normalized_class_dict,
            "normalized_class_dict_sha256": sha256(args.normalized_class_dict),
            "image_dir": args.image_dir,
        },
        "outputs": {
            "detection_summary": str(out_dir / "ftable_detection_summary.csv"),
            "dense_semantic_summary": str(out_dir / "ftable_dense_semantic_summary.csv"),
            "embedding_norms": str(out_dir / "ftable_embedding_norms.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    by_intervention = {}
    for row in detection_rows:
        by_intervention.setdefault(row["intervention"], []).append(row)
    dense_by_intervention = {}
    for row in dense_rows:
        dense_by_intervention.setdefault(row["intervention"], []).append(row)
    report = report_dir / "fres_p0148_support_intervention_reproduction.md"
    lines = [
        "# P0148 Support Intervention Reproduction",
        "",
        "## Scope",
        "",
        "本实验仅用于复现 P0148 病例上的旋转条件语义干扰，不参与 scene-disjoint N0-RI 的阈值、显著性检验或是否训练 RISC 的判定。",
        "",
        "## Mean Metrics",
        "",
        "| intervention | post_sv_ratio | post_lv_ratio | dense_top1_sv_ratio | dense_sv_score | box_mean_abs_delta | box_max_abs_delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for intervention in args.interventions:
        det = by_intervention[intervention]
        dense = dense_by_intervention[intervention]
        lines.append(
            "| {name} | {sv:.6f} | {lv:.6f} | {dense_sv:.6f} | {score:.6f} | {box_mean:.6f} | {box_max:.6f} |".format(
                name=intervention,
                sv=np.mean([row["small_vehicle_ratio"] for row in det]),
                lv=np.mean([row["large_vehicle_ratio"] for row in det]),
                dense_sv=np.mean([row["dense_top1_small_vehicle_ratio"] for row in dense]),
                score=np.mean([row["small_vehicle_score_mean"] for row in dense]),
                box_mean=np.mean([row["decoded_box_mean_abs_delta_vs_original"] for row in dense]),
                box_max=np.max([row["decoded_box_max_abs_delta_vs_original"] for row in dense]),
            )
        )
    lines.extend([
        "",
        "## Decision Boundary",
        "",
        "若 zero/swap/random 在 dense pre-NMS 指标上同步改变 small_vehicle 吸引强度，而 decoded box delta 接近零，则支持‘语义打分阶段已存在干扰，NMS 不是根因’。这仍然只是病例级因果诊断；RISC 的群体现象主张必须由独立的 scene-disjoint N0-RI 决定。",
        "",
        "## Outputs",
        "",
        f"- manifest: `{out_dir / 'manifest.json'}`",
        f"- detection_summary: `{out_dir / 'ftable_detection_summary.csv'}`",
        f"- dense_semantic_summary: `{out_dir / 'ftable_dense_semantic_summary.csv'}`",
        f"- embedding_norms: `{out_dir / 'ftable_embedding_norms.csv'}`",
        "",
    ])
    report.write_text("\n".join(lines))
    print(f"manifest={out_dir / 'manifest.json'}")
    print(f"report={report}")


if __name__ == "__main__":
    main()

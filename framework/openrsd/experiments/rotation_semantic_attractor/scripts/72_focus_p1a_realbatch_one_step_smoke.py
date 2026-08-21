#!/usr/bin/env python3
"""Run one real detector training step for FOCUS P1A target-mask injection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

from focus_p1a_realbatch_common import (
    DOTA2_CLASSES,
    ensure_exp_tree,
    parse_dota_ann,
    qbox_to_cxcywha,
    read_json,
    resolve,
    write_csv,
    write_json,
    write_manifest,
)


DEFAULT_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_sv_only_dota2_recovery_full.py")
DEFAULT_CKPT = Path(
    "results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth")
DEFAULT_EXP = Path("resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FOCUS P1A Realbatch One-Step Smoke",
        "",
        f"- status: `{payload.get('status')}`",
        f"- actual_detector_train: `{payload.get('actual_detector_train')}`",
        f"- real_batch_source: `{payload.get('real_batch_source')}`",
        f"- num_focus_anti_targets: `{payload.get('num_focus_anti_targets')}`",
        f"- num_focus_preserve_targets: `{payload.get('num_focus_preserve_targets')}`",
        f"- num_focus_anti_points: `{payload.get('num_focus_anti_points')}`",
        f"- num_focus_preserve_points: `{payload.get('num_focus_preserve_points')}`",
        f"- loss_focus_anti: `{payload.get('loss_focus_anti')}`",
        f"- loss_focus_text_anchor: `{payload.get('loss_focus_text_anchor')}`",
        f"- loss_focus_preserve: `{payload.get('loss_focus_preserve')}`",
        f"- loss_focus_total: `{payload.get('loss_focus_total')}`",
        f"- adapter_grad_flow: `{payload.get('adapter_grad_flow')}`",
        f"- frozen_grad_present: `{payload.get('frozen_grad_present')}`",
        f"- adapter_checksum_changed_after_step: `{payload.get('adapter_checksum_changed_after_step')}`",
        "",
    ]
    if payload.get("error"):
        lines.extend(["## Error", "", "```text", payload["error"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def checksum_params(named_params) -> str:
    h = hashlib.sha256()
    for name, param in named_params:
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def grad_summary(model, trainable_substrings: list[str] | None = None) -> tuple[list[dict[str, Any]], bool, bool]:
    rows: list[dict[str, Any]] = []
    adapter_grad_flow = False
    frozen_grad_present = False
    trainable_substrings = trainable_substrings or ["bbox_head.focus_support_adapter"]
    for name, param in model.named_parameters():
        grad_norm = 0.0
        has_grad = param.grad is not None
        if has_grad:
            grad_norm = float(param.grad.detach().pow(2).sum().sqrt().item())
        is_adapter = any(key in name for key in trainable_substrings)
        if is_adapter and has_grad and grad_norm > 0:
            adapter_grad_flow = True
        if (not param.requires_grad) and has_grad and grad_norm > 0:
            frozen_grad_present = True
        if is_adapter or has_grad:
            rows.append({
                "parameter": name,
                "requires_grad": str(bool(param.requires_grad)).lower(),
                "is_focus_adapter": str(bool(is_adapter)).lower(),
                "has_grad": str(bool(has_grad)).lower(),
                "grad_norm": f"{grad_norm:.10f}",
            })
    return rows, adapter_grad_flow, frozen_grad_present


def scalar_loss_rows(losses: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    import torch

    rows: list[dict[str, Any]] = []
    total = None
    for key, value in losses.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for idx, item in enumerate(values):
            if not torch.is_tensor(item):
                continue
            scalar = item.mean()
            total = scalar if total is None else total + scalar
            rows.append({
                "loss_key": key if len(values) == 1 else f"{key}_{idx}",
                "value": f"{float(scalar.detach().item()):.10f}",
                "finite": str(bool(torch.isfinite(scalar.detach()).item())).lower(),
            })
    if total is None:
        raise RuntimeError("no tensor losses produced by detector")
    return rows, total


class RealSmokeDataset:
    def __init__(self, selected_images: list[dict[str, Any]], img_scale: tuple[int, int]):
        self.selected_images = selected_images
        self.img_scale = img_scale

    def __len__(self) -> int:
        return len(self.selected_images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image
        import numpy as np
        import torch
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        from mmrotate.structures import RotatedBoxes

        item = self.selected_images[idx]
        image_path = Path(item["image_path"])
        image = Image.open(image_path).convert("RGB")
        ori_w, ori_h = image.size
        target_w, target_h = self.img_scale
        scale = min(target_w / ori_w, target_h / ori_h)
        new_w = int(round(ori_w * scale))
        new_h = int(round(ori_h * scale))
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
        canvas.paste(resized, (0, 0))
        arr_rgb = np.asarray(canvas, dtype=np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()
        inputs = torch.from_numpy(arr_bgr.transpose(2, 0, 1)).float()

        ann_objects = parse_dota_ann(Path(item["annotation_path"]))
        qboxes = []
        texts = []
        for obj in ann_objects:
            class_name = obj["class_name"]
            if class_name not in DOTA2_CLASSES:
                continue
            qbox = []
            for q_idx, value in enumerate(obj["qbox"]):
                qbox.append(float(value) * scale)
            qboxes.append(qbox)
            texts.append(class_name)
        if not qboxes:
            raise RuntimeError(f"no supported DOTA objects in {item['annotation_path']}")
        rboxes = torch.tensor(
            [qbox_to_cxcywha(qbox) for qbox in qboxes],
            dtype=torch.float32)
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(rboxes)
        gt_instances.texts = list(texts)
        gt_instances.visual_embeds = torch.zeros((len(texts), 1024), dtype=torch.float32)
        gt_instances.text_embeds = torch.zeros((len(texts), 768), dtype=torch.float32)

        sample = DetDataSample()
        sample.gt_instances = gt_instances
        sample.cls_list = list(DOTA2_CLASSES)
        sample.set_metainfo({
            "img_path": str(image_path),
            "img_id": item["tile_id"],
            "tile_id": item["tile_id"],
            "angle": int(item["angle"]),
            "ori_shape": (ori_h, ori_w),
            "img_shape": (target_h, target_w),
            "pad_shape": (target_h, target_w),
            "scale_factor": (scale, scale),
            "flip": False,
            "dataset_flag": "Data1_DOTA2",
        })
        return {
            "inputs": inputs,
            "data_sample": sample,
            "scale": scale,
            "annotation_object_count": len(texts),
        }


def run_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import numpy as np
    import torch
    from mmengine.config import Config, ConfigDict
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmrotate.registry import MODELS

    import M_AD.models.detectors.Flex_Rtmdet_v3_1_formal  # noqa: F401
    import M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1  # noqa: F401

    smoke_dataset = read_json(args.smoke_dataset_json, {})
    selected_images = smoke_dataset.get("selected_images", [])
    if not selected_images:
        raise RuntimeError("smoke dataset has no selected_images")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_default_scope("mmrotate")

    cfg = Config.fromfile(str(args.focus_config))
    cfg.model.backbone.init_cfg = None
    cfg.model.use_declip_support = False
    cfg.model.with_aux_bbox_head = False
    cfg.model.with_image_rec_losses = False
    cfg.model.bbox_head.with_obj_align = False
    variant_payload: dict[str, Any] = {}
    trainable_substrings = ["bbox_head.focus_support_adapter"]
    if args.variant_config_json:
        variant_payload = read_json(args.variant_config_json, {})
        cfg.model.support_type = variant_payload.get("support_type", "visual")
        cfg.model.bbox_head.use_focus_ovd = bool(
            variant_payload.get("use_focus_ovd", True))
        cfg.model.bbox_head.focus_ovd = ConfigDict(
            variant_payload.get("focus_ovd", {}))
        cfg.model.bbox_head.focus_losses = ConfigDict(
            variant_payload.get("focus_losses", {}))
        trainable_substrings = list(
            variant_payload.get("trainable_substrings", trainable_substrings))
    else:
        cfg.model.support_type = "visual"
        cfg.model.bbox_head.use_focus_ovd = True
        cfg.model.bbox_head.focus_ovd = ConfigDict(
            enable=True,
            orientation=ConfigDict(
                patch_size=7,
                num_angle_bins=36,
                harmonic_orders=(2, 4, 6),
                detach_orientation=True),
            adapter=ConfigDict(
                apply_to_classes=("small-vehicle",),
                alpha_init=0.05,
                alpha_max=0.10,
                max_delta_norm_ratio=0.05,
                low_rank=16))
        cfg.model.bbox_head.focus_losses = ConfigDict(
            enable=True,
            target_mapping_mode="spatial_region",
            spatial_target_csv=str(args.spatial_targets),
            support_distill_weight=0.01,
            anti_attractor_weight=0.05,
            preserve_weight=0.05,
            migration_weight=0.0,
            anti_margin=0.10,
            max_focus_points_per_image=256,
            max_points_per_target=16)

    model = MODELS.build(cfg.model)
    checkpoint_path = args.checkpoint
    if checkpoint_path.exists():
        load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for SyncBatchNorm smoke")
    if not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{free_tcp_port()}",
            rank=0,
            world_size=1)
    model.to(device)
    if (not variant_payload
            and getattr(model.bbox_head, "focus_support_adapter", None) is not None):
        with torch.no_grad():
            model.bbox_head.focus_support_adapter.alpha.fill_(0.05)
    for name, param in model.named_parameters():
        param.requires_grad_(any(key in name for key in trainable_substrings))
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable focus adapter parameters")
    optimizer = torch.optim.SGD(trainable, lr=args.lr)
    model.train()

    dataset = RealSmokeDataset(selected_images, img_scale=(832, 832))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch)
    batch = next(iter(loader))
    inputs = [item["inputs"] for item in batch]
    data_samples = [item["data_sample"] for item in batch]
    data = model.data_preprocessor({
        "inputs": inputs,
        "data_samples": data_samples,
    }, training=True)
    focus_named = [
        (name, param) for name, param in model.named_parameters()
        if any(key in name for key in trainable_substrings)
    ]
    checksum_before = checksum_params(focus_named)
    loss_history: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    adapter_grad_flow = False
    adapter_grad_flow_any = False
    frozen_grad_present = False
    frozen_grad_present_any = False
    loss_rows: list[dict[str, Any]] = []
    max_iters = max(1, int(args.max_iters))
    checkpoints = set([1, max_iters])
    checkpoints.update(step for step in range(100, max_iters + 1, 100))
    for step in range(1, max_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(data["inputs"], data["data_samples"])
        loss_rows, total_loss = scalar_loss_rows(losses)
        total_loss.backward()
        grad_rows, adapter_grad_flow, frozen_grad_present = grad_summary(
            model, trainable_substrings)
        adapter_grad_flow_any = adapter_grad_flow_any or bool(adapter_grad_flow)
        frozen_grad_present_any = (
            frozen_grad_present_any or bool(frozen_grad_present))
        optimizer.step()
        if step in checkpoints:
            loss_history.append({
                "step": step,
                "loss_total": float(total_loss.detach().item()),
                **{
                    row["loss_key"]: float(row["value"])
                    for row in loss_rows
                },
            })
    checksum_after = checksum_params(focus_named)

    focus_loss_debug = getattr(model.bbox_head, "_last_focus_loss_debug", {}) or {}
    assignment_debug = getattr(model, "_last_focus_assignment_debug", {}) or {}
    support_delta_norm_ratios = []
    text_delta_norm_ratios = []
    text_support_cos_max = []
    dual_text_weights = []
    for debug in getattr(model.bbox_head, "_last_focus_debug_by_level", []) or []:
        if not debug:
            continue
        adapter_debug = debug.get("adapter") or {}
        ratio = adapter_debug.get("delta_norm_ratio")
        if ratio is not None and hasattr(ratio, "detach"):
            support_delta_norm_ratios.append(float(ratio.detach().max().item()))
        eqtext_debug = debug.get("eqtext") or {}
        text_ratio = eqtext_debug.get("text_delta_norm_ratio")
        if text_ratio is not None and hasattr(text_ratio, "detach"):
            text_delta_norm_ratios.append(float(text_ratio.detach().max().item()))
        text_cos = eqtext_debug.get("text_interclass_cos_max")
        if text_cos is not None and hasattr(text_cos, "detach"):
            text_support_cos_max.append(float(text_cos.detach().max().item()))
        fusion_debug = debug.get("dual_fusion") or {}
        if "dual_text_weight" in fusion_debug:
            dual_text_weights.append(float(fusion_debug.get("dual_text_weight", 0.0)))
    support_delta_norm_ratio = max(support_delta_norm_ratios) if support_delta_norm_ratios else 0.0
    text_delta_norm_ratio = max(text_delta_norm_ratios) if text_delta_norm_ratios else 0.0
    text_cos_max = max(text_support_cos_max) if text_support_cos_max else 0.0
    dual_text_weight = max(dual_text_weights) if dual_text_weights else 0.0

    loss_map = {
        row["loss_key"]: float(row["value"])
        for row in loss_rows if row["loss_key"].startswith("loss_focus_")
    }
    actual_detector_train = True
    pass_conditions = {
        "actual_detector_train": actual_detector_train,
        "real_batch_source": "dataloader_real_batch",
        "num_focus_images_with_targets": int(focus_loss_debug.get(
            "num_focus_images_with_targets", 0)) >= 1,
        "num_focus_anti_targets": int(focus_loss_debug.get(
            "num_focus_anti_targets", 0)) >= 1,
        "num_focus_preserve_targets": int(focus_loss_debug.get(
            "num_focus_preserve_targets", 0)) >= 1,
        "num_focus_anti_points": int(focus_loss_debug.get(
            "num_focus_anti_points", 0)) > 0,
        "num_focus_preserve_points": int(focus_loss_debug.get(
            "num_focus_preserve_points", 0)) > 0,
        "loss_focus_anti": float(loss_map.get("loss_focus_anti", 0.0)) > 0.0,
        "loss_focus_preserve": float(loss_map.get("loss_focus_preserve", 0.0)) > 0.0,
        "loss_focus_total_finite": all(
            row["finite"] == "true" for row in loss_rows
            if row["loss_key"].startswith("loss_focus_")),
        "adapter_grad_flow": bool(adapter_grad_flow_any),
        "frozen_grad_present_false": not bool(frozen_grad_present_any),
        "adapter_checksum_changed_after_step": checksum_before != checksum_after,
        "declip_support_disabled": not bool(getattr(model, "use_declip_support", False)),
        "support_delta_norm_ratio_le_0_05": support_delta_norm_ratio <= 0.050001,
        "dual_text_weight_le_0_2": dual_text_weight <= 0.200001,
    }
    status = (
        "PASS_P1A_REALBATCH_ONE_STEP"
        if all(pass_conditions.values())
        else "FAIL_ANTI_NOT_ACTIVE"
        if not pass_conditions["loss_focus_anti"]
        else "FAIL_P1A_REALBATCH_ONE_STEP")
    payload = {
        "status": status,
        "seed": args.seed,
        "variant_id": variant_payload.get("variant_id", "P1A_DEFAULT"),
        "max_iters": max_iters,
        "actual_detector_train": actual_detector_train,
        "actual_detector_rerun": False,
        "real_batch_source": "dataloader_real_batch",
        "checkpoint": str(checkpoint_path),
        "focus_config": str(args.focus_config),
        "spatial_targets": str(args.spatial_targets),
        "smoke_dataset_json": str(args.smoke_dataset_json),
        "selected_images": selected_images,
        "batch_targets": assignment_debug.get("lookup_debug_rows", []),
        "focus_assignment_rows": assignment_debug.get("debug_rows", []),
        "focus_assignment_debug": assignment_debug,
        "focus_loss_debug": focus_loss_debug,
        "num_focus_images_with_targets": int(focus_loss_debug.get(
            "num_focus_images_with_targets", 0)),
        "num_focus_anti_targets": int(focus_loss_debug.get(
            "num_focus_anti_targets", 0)),
        "num_focus_preserve_targets": int(focus_loss_debug.get(
            "num_focus_preserve_targets", 0)),
        "num_focus_anti_points": int(focus_loss_debug.get(
            "num_focus_anti_points", 0)),
        "num_focus_preserve_points": int(focus_loss_debug.get(
            "num_focus_preserve_points", 0)),
        "focus_assignment_coverage": float(focus_loss_debug.get(
            "focus_assignment_coverage", 0.0)),
        "loss_focus_support_distill": float(loss_map.get("loss_focus_support_distill", 0.0)),
        "loss_focus_text_anchor": float(loss_map.get("loss_focus_text_anchor", 0.0)),
        "loss_focus_anti": float(loss_map.get("loss_focus_anti", 0.0)),
        "loss_focus_preserve": float(loss_map.get("loss_focus_preserve", 0.0)),
        "loss_focus_migration": float(loss_map.get("loss_focus_migration", 0.0)),
        "loss_focus_total": float(loss_map.get("loss_focus_total", 0.0)),
        "adapter_grad_flow": bool(adapter_grad_flow_any),
        "adapter_grad_flow_last_step": bool(adapter_grad_flow),
        "frozen_grad_present": bool(frozen_grad_present_any),
        "frozen_grad_present_last_step": bool(frozen_grad_present),
        "adapter_checksum_before": checksum_before,
        "adapter_checksum_after": checksum_after,
        "adapter_checksum_changed_after_step": checksum_before != checksum_after,
        "declip_support_disabled": not bool(getattr(model, "use_declip_support", False)),
        "native_support_preserved": True,
        "support_delta_norm_ratio": support_delta_norm_ratio,
        "text_delta_norm_ratio": text_delta_norm_ratio,
        "text_support_cos_max": text_cos_max,
        "dual_text_weight": dual_text_weight,
        "trainable_substrings": trainable_substrings,
        "loss_history": loss_history,
        "no_nan_inf": all(row["finite"] == "true" for row in loss_rows),
        "pass_conditions": pass_conditions,
        "evidence_level": "DETECTOR_LEVEL_SPATIAL_PSEUDO_REGION_REAL_BATCH",
    }
    write_csv(output_dir / "realbatch_loss_values.csv", loss_rows,
              ["loss_key", "value", "finite"])
    write_csv(output_dir / "realbatch_loss_history.csv",
              loss_history,
              sorted({key for row in loss_history for key in row.keys()}))
    write_csv(output_dir / "realbatch_grad_flow.csv", grad_rows,
              ["parameter", "requires_grad", "is_focus_adapter", "has_grad", "grad_norm"])
    assignment_fields = [
        "batch_index",
        "img_path",
        "parsed_tile_id",
        "parsed_angle",
        "focus_target_id",
        "level",
        "grid_x",
        "grid_y",
        "point_x",
        "point_y",
        "inside_polygon",
        "assigned_role",
        "conflict_resolution",
        "valid",
    ]
    write_csv(output_dir / "realbatch_focus_assignment.csv",
              assignment_debug.get("debug_rows", []), assignment_fields)
    write_json(output_dir / "realbatch_adapter_checksum.json", {
        "before": checksum_before,
        "after": checksum_after,
        "changed": checksum_before != checksum_after,
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spatial-targets", type=Path, required=True)
    parser.add_argument("--smoke-dataset-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP / "one_batch_real")
    parser.add_argument("--focus-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--variant-config-json", type=Path)
    parser.add_argument("--max-iters", type=int, default=1)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    args.spatial_targets = resolve(repo_root, args.spatial_targets)
    args.smoke_dataset_json = resolve(repo_root, args.smoke_dataset_json)
    args.output_dir = resolve(repo_root, args.output_dir)
    args.focus_config = resolve(repo_root, args.focus_config)
    args.checkpoint = resolve(repo_root, args.checkpoint)
    if args.variant_config_json is not None:
        args.variant_config_json = resolve(repo_root, args.variant_config_json)
    exp_dir = args.output_dir.parent
    ensure_exp_tree(exp_dir)
    write_manifest(exp_dir, repo_root, {
        "spatial_targets": str(args.spatial_targets),
        "smoke_dataset_json": str(args.smoke_dataset_json),
    })

    try:
        payload = run_smoke(args, args.output_dir)
    except Exception as exc:
        payload = {
            "status": "FAIL_P1A_REALBATCH_ONE_STEP_EXCEPTION",
            "actual_detector_train": False,
            "real_batch_source": "dataloader_real_batch",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "num_focus_anti_targets": 0,
            "num_focus_preserve_targets": 0,
            "num_focus_anti_points": 0,
            "num_focus_preserve_points": 0,
            "loss_focus_anti": 0.0,
            "loss_focus_preserve": 0.0,
            "loss_focus_total": 0.0,
            "adapter_grad_flow": False,
            "frozen_grad_present": None,
            "adapter_checksum_changed_after_step": False,
            "evidence_level": "INSUFFICIENT_P1A_REAL_BATCH_EVIDENCE",
            "spatial_targets": str(args.spatial_targets),
            "smoke_dataset_json": str(args.smoke_dataset_json),
        }
    write_json(args.output_dir / "realbatch_smoke_report.json", payload)
    write_markdown(args.output_dir / "realbatch_smoke_report.md", payload)
    print(json.dumps({
        "status": payload.get("status"),
        "actual_detector_train": payload.get("actual_detector_train"),
        "anti_points": payload.get("num_focus_anti_points"),
        "preserve_points": payload.get("num_focus_preserve_points"),
        "loss_focus_anti": payload.get("loss_focus_anti"),
        "loss_focus_preserve": payload.get("loss_focus_preserve"),
    }, indent=2))
    return 0 if str(payload.get("status", "")).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

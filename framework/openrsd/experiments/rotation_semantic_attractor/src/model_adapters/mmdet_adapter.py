from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..constants import DOTA1_CLASSES, SMALL_VEHICLE
from ..geometry import ensure_polygon
from .capabilities import build_capability_profile
from .base import BaseDetectorAdapter


SMALL_VEHICLE_ID = DOTA1_CLASSES.index(SMALL_VEHICLE)


class MMDetAdapter(BaseDetectorAdapter):
    def __init__(self, model_key: str, cfg: Dict[str, Any], project_root: str | Path):
        super().__init__(model_key, cfg, project_root)
        self.model = None
        self.device = "cpu"
        self.capability_profile = build_capability_profile(model_key, cfg)

    def load(self, device: str):
        ok, reason = self.available()
        if not ok:
            raise FileNotFoundError(reason)
        from mmdet.apis import init_detector

        self.device = device
        self.model = init_detector(self.cfg["config"], self.cfg["checkpoint"], device=device)
        return self

    def _tensor_to_numpy(self, value):
        if hasattr(value, "tensor"):
            value = value.tensor
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _prediction_rows(self, pred_instances, tile_id: str, angle: int, stage: str) -> List[Dict[str, Any]]:
        bboxes = self._tensor_to_numpy(pred_instances.bboxes)
        scores = self._tensor_to_numpy(pred_instances.scores)
        if scores.ndim == 2:
            labels = scores.argmax(axis=1).astype(int)
            scores = scores.max(axis=1)
        elif hasattr(pred_instances, "labels"):
            labels = self._tensor_to_numpy(pred_instances.labels).astype(int)
        else:
            labels = np.zeros((len(scores),), dtype=int)
        rows = []
        for idx, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            label_int = int(label)
            class_name = DOTA1_CLASSES[label_int] if 0 <= label_int < len(DOTA1_CLASSES) else str(label_int)
            box_type = "obb" if len(bbox) == 5 else "polygon" if len(bbox) >= 8 else "hbb"
            box = [float(v) for v in bbox.tolist()]
            rows.append(
                {
                    "box": box,
                    "polygon": ensure_polygon(box, box_type),
                    "box_type": box_type,
                    "score": float(score),
                    "class_id": label_int,
                    "class_name": class_name,
                    "stage": stage,
                    "raw_index": idx,
                    "angle": int(angle),
                    "tile_id": tile_id,
                    "model_name": self.model_key,
                }
            )
        return rows

    def _prepare_data(self, image_path: str | Path):
        from mmcv.transforms import Compose
        from mmdet.utils import get_test_pipeline_cfg

        cfg = self.model.cfg.copy()
        test_pipeline = Compose(get_test_pipeline_cfg(cfg))
        data = test_pipeline({"img_path": str(image_path), "img_id": 0})
        data["inputs"] = [data["inputs"]]
        data["data_samples"] = [data["data_samples"]]
        return data

    def _dense_logits_summary(self, outs) -> Dict[str, float] | None:
        if not outs:
            return None
        cls_scores = outs[0]
        if not isinstance(cls_scores, (list, tuple)):
            return None
        values = []
        for cls_score in cls_scores:
            if not hasattr(cls_score, "detach") or cls_score.ndim != 4:
                continue
            score = cls_score.detach().float().sigmoid().cpu()
            batch, channels, height, width = score.shape
            num_classes = len(DOTA1_CLASSES)
            if channels % num_classes == 0:
                num_anchors = channels // num_classes
                score = score.reshape(batch, num_anchors, num_classes, height, width)[:, :, SMALL_VEHICLE_ID]
            elif channels > SMALL_VEHICLE_ID:
                score = score[:, SMALL_VEHICLE_ID : SMALL_VEHICLE_ID + 1]
            else:
                continue
            values.append(score.reshape(-1))
        if not values:
            return None
        import torch

        merged = torch.cat(values)
        topk = merged.topk(min(1000, merged.numel())).values
        return {
            "small_vehicle_mean_score": float(merged.mean().item()),
            "small_vehicle_max_score": float(merged.max().item()),
            "small_vehicle_topk_mean_score": float(topk.mean().item()),
            "num_dense_scores": int(merged.numel()),
        }

    def _limit_predictions(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        limit = int(self.cfg.get("max_pre_nms_predictions", 2000))
        if limit <= 0 or len(rows) <= limit:
            return rows
        return sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)[:limit]

    def _infer_pre_nms_and_dense(self, image_path: str | Path, tile_id: str, angle: int) -> tuple[List[Dict[str, Any]], Dict[str, float] | None]:
        if not hasattr(self.model, "bbox_head"):
            raise NotImplementedError("model_has_no_single_stage_bbox_head")
        data = self._prepare_data(image_path)
        with __import__("torch").no_grad():
            processed = self.model.data_preprocessor(data, False)
            feats = self.model.extract_feat(processed["inputs"])
            outs = self.model.bbox_head(feats)
            batch_img_metas = [sample.metainfo for sample in processed["data_samples"]]
            pre_nms = self.model.bbox_head.predict_by_feat(
                *outs,
                batch_img_metas=batch_img_metas,
                rescale=True,
                with_nms=False,
            )[0]
        rows = self._prediction_rows(pre_nms, tile_id=tile_id, angle=angle, stage="pre_nms")
        return self._limit_predictions(rows), self._dense_logits_summary(outs)

    def infer(self, image_path: str | Path, tile_id: str, angle: int, prompts=None, return_raw: bool = True) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("adapter not loaded")
        from mmdet.apis import inference_detector

        result = inference_detector(self.model, str(image_path))
        final_predictions = self._prediction_rows(result.pred_instances, tile_id=tile_id, angle=angle, stage="final")
        unsupported = []
        pre_nms_predictions = []
        dense_logits = None
        if self.info.supports_pre_nms or self.info.supports_dense_logits:
            try:
                pre_nms_predictions, dense_logits = self._infer_pre_nms_and_dense(image_path, tile_id=tile_id, angle=angle)
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                if self.info.supports_pre_nms:
                    unsupported.append({"stage": "pre_nms", "reason": f"pre_nms_hook_failed:{reason}"})
                if self.info.supports_dense_logits:
                    unsupported.append({"stage": "dense", "reason": f"dense_logits_hook_failed:{reason}"})
        if not self.info.supports_pre_nms:
            unsupported.append({"stage": "pre_nms", "reason": "pre_nms_hook_not_implemented"})
        if not self.info.supports_dense_logits:
            unsupported.append({"stage": "dense", "reason": "dense_logits_not_available"})
        return {
            "final_predictions": final_predictions,
            "pre_nms_predictions": pre_nms_predictions,
            "dense_logits": dense_logits,
            "query_logits": None,
            "region_features": None,
            "class_embeddings": None,
            "metadata": {
                "adapter": "mmdet",
                "device": self.device,
                "config": self.cfg.get("config", ""),
                "checkpoint": self.cfg.get("checkpoint", ""),
                "capability_profile": self.capability_profile,
                "unsupported": unsupported,
            },
        }

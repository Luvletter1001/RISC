from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class DetectorAdapterInfo:
    name: str
    model_type: str
    box_type: str
    supports_dense_logits: bool
    supports_query_logits: bool
    supports_pre_nms: bool
    supports_embedding_intervention: bool
    supports_classifier_channel_intervention: bool


class BaseDetectorAdapter:
    def __init__(self, model_key: str, cfg: Dict[str, Any], project_root: str | Path):
        self.model_key = model_key
        self.cfg = cfg
        self.project_root = Path(project_root)
        self.info = DetectorAdapterInfo(
            name=cfg.get("name", model_key),
            model_type=cfg.get("model_type", "unknown"),
            box_type=cfg.get("box_type", "unknown"),
            supports_dense_logits=bool(cfg.get("supports_dense_logits", False)),
            supports_query_logits=bool(cfg.get("supports_query_logits", False)),
            supports_pre_nms=bool(cfg.get("supports_pre_nms", False)),
            supports_embedding_intervention=bool(cfg.get("supports_embedding_intervention", False)),
            supports_classifier_channel_intervention=bool(cfg.get("supports_classifier_channel_intervention", False)),
        )

    def available(self) -> tuple[bool, str]:
        config = self.cfg.get("config", "")
        checkpoint = self.cfg.get("checkpoint", "")
        if config and not Path(config).exists():
            return False, f"missing_config:{config}"
        if checkpoint and not Path(checkpoint).exists():
            return False, f"missing_checkpoint:{checkpoint}"
        if not checkpoint:
            return False, "missing_checkpoint"
        return True, ""

    def load(self, device: str):
        raise NotImplementedError

    def infer(self, image_path: str | Path, tile_id: str, angle: int, prompts=None, return_raw: bool = True) -> Dict[str, Any]:
        raise NotImplementedError


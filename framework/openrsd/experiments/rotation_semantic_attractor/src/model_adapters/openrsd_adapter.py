from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from .base import BaseDetectorAdapter


@dataclass
class OpenRSDRuntime:
    base_args: Any
    model: Any
    device: Any
    support_data: Any
    name2id: dict
    id2name: dict
    loader: Any | None = None


class OpenRSDAdapter(BaseDetectorAdapter):
    """Thin adapter around the local OpenRSD/MMRotate diagnostic runtime.

    Heavy imports stay inside `load` so regular unit tests can import this
    adapter without requiring the OpenRSD conda environment.
    """

    def __init__(self, model_key: str, cfg: Dict[str, Any], project_root: str | Path):
        super().__init__(model_key, cfg, project_root)
        self.runtime: OpenRSDRuntime | None = None
        self.device = "cpu"

    def available(self) -> tuple[bool, str]:
        ok, reason = super().available()
        if not ok:
            return ok, reason
        support = self.cfg.get("support_pkl") or self.cfg.get("support_feat")
        if support and not Path(support).exists():
            return False, f"missing_support_pkl:{support}"
        return True, ""

    def _base_args(self, *, image_dir: str | Path | None, out_dir: str | Path | None, angles: list[int] | None):
        return SimpleNamespace(
            config=str(self.cfg["config"]),
            checkpoint=str(self.cfg["checkpoint"]),
            image_dir=str(image_dir or ""),
            out_dir=str(out_dir or self.project_root / "work_dirs/openrsd_adapter"),
            angles=list(angles or []),
            angle_step=None,
            score_thr=float(self.cfg.get("score_thr", 0.3)),
            iou_thr=float(self.cfg.get("iou_thr", 0.5)),
            support_shot=int(self.cfg.get("support_shot", 8)),
            support_type=str(self.cfg.get("support_type", "visual")),
            support_feat=str(self.cfg.get("support_pkl") or self.cfg.get("support_feat") or ""),
            normalized_class_dict=str(self.cfg.get("normalized_class_dict", "data/normalized_class_dict.pkl")),
            batch_size=int(self.cfg.get("batch_size", 1)),
            num_workers=int(self.cfg.get("num_workers", 0)),
            seed=int(self.cfg.get("seed", 20260530)),
            device=str(self.cfg.get("device", "cuda:0")),
            max_spatial_size=int(self.cfg.get("max_spatial_size", 32)),
            result_md_dir=str(self.cfg.get("result_md_dir", "resultmd/openrsd_adapter")),
        )

    def load(
        self,
        device: str = "cuda:0",
        *,
        image_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
        angles: list[int] | None = None,
    ):
        ok, reason = self.available()
        if not ok:
            raise FileNotFoundError(reason)
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as base

        args = self._base_args(image_dir=image_dir, out_dir=out_dir, angles=angles)
        args.device = device
        self.device = device
        support_path, _ = base.resolve_support_path(args.support_feat)
        _, model, _ = base.build_runner_model(args, support_path)
        import torch

        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(torch_device).eval()
        _, _, support_data, name2id, id2name = base.prepare_support(args, torch_device)
        self.runtime = OpenRSDRuntime(args, model, torch_device, support_data, name2id, id2name)
        return self

    def build_loader(self, image_dir: str | Path, angles: list[int]):
        if self.runtime is None:
            raise RuntimeError("adapter not loaded")
        from tools.rotation_diagnostics import probe_rotated_stage_outputs as base

        self.runtime.base_args.image_dir = str(image_dir)
        self.runtime.base_args.angles = list(angles)
        self.runtime.loader = base.build_dataloader(self.runtime.base_args, angles)
        return self.runtime.loader

    def infer(self, image_path: str | Path, tile_id: str, angle: int, prompts=None, return_raw: bool = True):
        raise NotImplementedError("OpenRSD inference requires dataset samples; use scripts/13_openrsd_hook_smoke.py")

"""Log dehub loss and cap loss weight if ratio too high."""
from __future__ import annotations

import csv
from pathlib import Path

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _scalar(v):
    if v is None:
        return 0.0
    if hasattr(v, 'item'):
        return float(v.item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@HOOKS.register_module()
class SvDehubTrainHook(Hook):
    priority = 'NORMAL'

    def __init__(
        self,
        log_csv: str,
        log_interval: int = 100,
        max_dehub_ratio: float = 0.30,
        min_weight: float = 0.01,
    ):
        self.log_csv = Path(log_csv)
        self.log_interval = log_interval
        self.max_dehub_ratio = max_dehub_ratio
        self.min_weight = min_weight
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_csv.exists():
            with open(self.log_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'iter', 'total_loss', 'loss_cls', 'loss_bbox', 'loss_dehub',
                    'dehub_ratio', 'lr', 'status',
                ])

    def _extract_losses(self, runner, outputs=None, **kwargs) -> dict:
        out = {}
        outputs = outputs if outputs is not None else kwargs.get('outputs')
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if 'loss' in str(k).lower():
                    out[k] = _scalar(v)
        if out.get('loss', 0) <= 0:
            lp = getattr(runner, 'log_processor', None)
            if lp is not None and getattr(lp, 'log_vars', None):
                for k, v in lp.log_vars.items():
                    out[k] = _scalar(v)
        return out

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, **kwargs) -> None:
        it = runner.iter + 1
        if it % self.log_interval != 0:
            return
        lv = self._extract_losses(runner, **kwargs)
        total = lv.get('loss', 0.0)
        cls_l = lv.get('loss_cls', 0.0)
        bbox_l = lv.get('loss_bbox', 0.0)
        dehub = lv.get('loss_dehub', 0.0)
        ratio = dehub / total if total > 1e-8 else 0.0
        lr = 0.0
        try:
            lr = runner.optim_wrapper.param_groups[0]['lr']
        except Exception:
            pass
        status = 'ok'
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        head = getattr(model, 'bbox_head', None)
        if head is not None and ratio > self.max_dehub_ratio:
            w = getattr(head, 'sv_dehub_loss_weight', 0.05)
            head.sv_dehub_loss_weight = max(self.min_weight, w * 0.5)
            status = f'weight_capped->{head.sv_dehub_loss_weight:.4f}'
        with open(self.log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                it, total, cls_l, bbox_l, dehub, ratio, lr, status,
            ])

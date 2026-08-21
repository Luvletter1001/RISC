"""Gradient presence audit hook for short control runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _grad_norm(param: torch.nn.Parameter) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().float().norm().item())


@HOOKS.register_module()
class GradPresenceAuditHook(Hook):
    """Record trainable-parameter gradient presence after backward.

    The hook wraps ``runner.optim_wrapper.backward`` in ``before_run`` so the
    audit happens after backward and before optimizer step / zero-grad.
    """

    priority = 'LOW'

    def __init__(self,
                 trainable_substrings: Iterable[str],
                 log_path: str,
                 summary_path: str | None = None,
                 audit_iters: Iterable[int] = (1, 50),
                 fail_on_missing: bool = False):
        keys = tuple(str(key) for key in trainable_substrings if str(key))
        if not keys:
            raise ValueError(
                'trainable_substrings must contain at least one non-empty key.')
        self.trainable_substrings = keys
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path) if summary_path else None
        self.audit_iters = {int(item) for item in audit_iters}
        self.fail_on_missing = bool(fail_on_missing)
        self._orig_backward = None
        self._latest_payload = None

    def before_run(self, runner) -> None:
        self._orig_backward = runner.optim_wrapper.backward

        def wrapped_backward(loss, *args, **kwargs):
            result = self._orig_backward(loss, *args, **kwargs)
            step = int(runner.iter) + 1
            if step in self.audit_iters:
                self._audit(runner, step)
            return result

        runner.optim_wrapper.backward = wrapped_backward
        runner.logger.info(
            'GradPresenceAuditHook enabled for iters=%s substrings=%s',
            sorted(self.audit_iters), list(self.trainable_substrings))

    def _audit(self, runner, step: int) -> None:
        rank = get_rank()
        model = _unwrap_model(runner.model)
        group_stats = {
            key: dict(
                matched_params=0,
                nonzero_grad_params=0,
                total_grad_norm=0.0,
                max_grad_norm=0.0,
                sample_nonzero=[],
                sample_zero=[],
            )
            for key in self.trainable_substrings
        }
        total_trainable = 0
        total_nonzero = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            total_trainable += 1
            norm = _grad_norm(param)
            if norm > 0:
                total_nonzero += 1
            for key, stats in group_stats.items():
                if key not in name:
                    continue
                stats['matched_params'] += 1
                stats['total_grad_norm'] += norm
                stats['max_grad_norm'] = max(stats['max_grad_norm'], norm)
                if norm > 0:
                    stats['nonzero_grad_params'] += 1
                    if len(stats['sample_nonzero']) < 8:
                        stats['sample_nonzero'].append(
                            dict(name=name, grad_norm=norm))
                elif len(stats['sample_zero']) < 8:
                    stats['sample_zero'].append(name)

        missing = [
            key for key, stats in group_stats.items()
            if stats['matched_params'] == 0 or stats['nonzero_grad_params'] == 0
        ]
        payload = dict(
            iter=step,
            rank=rank,
            total_trainable_params=total_trainable,
            total_nonzero_grad_params=total_nonzero,
            missing_grad_substrings=missing,
            ok=(len(missing) == 0),
            groups=group_stats,
        )
        self._latest_payload = payload

        if rank == 0:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            if self.summary_path is not None:
                self.summary_path.parent.mkdir(parents=True, exist_ok=True)
                self.summary_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')

        runner.logger.info(
            'GradPresenceAuditHook iter=%d ok=%s missing=%s '
            'nonzero_trainable=%d/%d',
            step, payload['ok'], missing, total_nonzero, total_trainable)
        if self.fail_on_missing and missing:
            raise RuntimeError(
                'Gradient audit found missing gradient substrings: '
                f'{missing}')

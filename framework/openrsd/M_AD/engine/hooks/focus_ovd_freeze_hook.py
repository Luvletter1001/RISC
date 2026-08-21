"""FOCUS-OVD trainability audit hooks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _matches_any(name: str, substrings: Iterable[str]) -> bool:
    return any(key in name for key in substrings)


@HOOKS.register_module()
class FocusOVDTrainableAuditHook(Hook):
    """Fail fast if FOCUS adapter-only training is not actually adapter-only."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 trainable_substrings: Iterable[str],
                 log_path: str | None = None,
                 fail_on_unexpected: bool = True):
        self.trainable_substrings = tuple(
            str(key) for key in trainable_substrings if str(key))
        if not self.trainable_substrings:
            raise ValueError(
                'trainable_substrings must contain at least one non-empty key.')
        self.log_path = Path(log_path) if log_path else None
        self.fail_on_unexpected = bool(fail_on_unexpected)

    def before_run(self, runner) -> None:
        model = _unwrap_model(runner.model)
        trainable: List[str] = []
        frozen: List[str] = []
        unexpected: List[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable.append(name)
                if not _matches_any(name, self.trainable_substrings):
                    unexpected.append(name)
            else:
                frozen.append(name)

        missing = [
            key for key in self.trainable_substrings
            if not any(key in name for name in trainable)
        ]
        payload = dict(
            trainable_substrings=list(self.trainable_substrings),
            trainable=trainable,
            frozen_count=len(frozen),
            unexpected_trainable=unexpected,
            missing_trainable_substrings=missing,
            ok=(not unexpected and not missing),
        )

        if self.log_path is not None and get_rank() == 0:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding='utf-8')

        if self.fail_on_unexpected and unexpected:
            raise RuntimeError(
                'FOCUS-OVD adapter-only audit found unexpected trainable '
                f'parameters: {unexpected[:20]}')
        if self.fail_on_unexpected and missing:
            raise RuntimeError(
                'FOCUS-OVD adapter-only audit found no trainable parameters '
                f'matching: {missing}')

        runner.logger.info(
            'FOCUS-OVD trainability audit passed with %d trainable '
            'parameters and %d frozen parameters.',
            len(trainable),
            len(frozen))

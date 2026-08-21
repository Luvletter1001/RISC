"""Hooks for freezing BatchNorm running statistics during train loops."""
from __future__ import annotations

from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class FreezeNormStatsHook(Hook):
    """Keep BatchNorm/SyncBatchNorm layers in eval mode during training.

    This freezes running_mean/running_var/num_batches_tracked while leaving
    affine parameters and the rest of the model untouched.
    """

    priority = 'VERY_HIGH'

    def __init__(self, log_each_epoch: bool = True):
        self.log_each_epoch = bool(log_each_epoch)
        self._num_norm_layers = None

    def _freeze_norm(self, runner) -> int:
        model = _unwrap_model(runner.model)
        count = 0
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()
                count += 1
        self._num_norm_layers = count
        return count

    def before_train(self, runner) -> None:
        count = self._freeze_norm(runner)
        runner.logger.info(
            'FreezeNormStatsHook enabled; froze %d BatchNorm-like layers.',
            count)

    def before_train_epoch(self, runner) -> None:
        count = self._freeze_norm(runner)
        if self.log_each_epoch:
            runner.logger.info(
                'FreezeNormStatsHook kept %d BatchNorm-like layers in eval '
                'mode for epoch %d.',
                count, runner.epoch + 1)

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._freeze_norm(runner)

from mmengine.hooks import CheckpointHook

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class MilestoneCheckpointHook(CheckpointHook):
    """Save epoch checkpoints only at an explicit milestone allowlist."""

    def __init__(self, milestones, **kwargs):
        values = tuple(milestones)
        if (not values or any(type(value) is not int or value <= 0
                              for value in values)
                or any(left >= right
                       for left, right in zip(values, values[1:]))):
            raise ValueError(
                'milestones must be strictly increasing positive integers')
        self.milestones = frozenset(values)
        super().__init__(
            interval=-1, by_epoch=True, save_last=False, **kwargs)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch in self.milestones:
            runner.logger.info('Saving milestone checkpoint at %d epochs',
                               epoch)
            self._save_checkpoint(runner)

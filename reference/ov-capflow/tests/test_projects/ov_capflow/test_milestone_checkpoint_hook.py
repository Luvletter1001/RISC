import pytest


class _Logger:
    def info(self, *args, **kwargs):
        pass


class _Runner:
    def __init__(self):
        self.epoch = 0
        self.logger = _Logger()


def test_milestone_checkpoint_hook_saves_only_requested_epochs(monkeypatch):
    from projects.OVCapFlow.ov_capflow.milestone_checkpoint_hook import (
        MilestoneCheckpointHook, )

    hook = MilestoneCheckpointHook(milestones=(1, 6, 12, 18, 24))
    runner = _Runner()
    saved = []
    monkeypatch.setattr(
        hook, '_save_checkpoint',
        lambda current_runner: saved.append(current_runner.epoch + 1))

    for epoch in range(24):
        runner.epoch = epoch
        hook.after_train_epoch(runner)

    assert saved == [1, 6, 12, 18, 24]


@pytest.mark.parametrize('milestones', [(), (0, 1), (6, 1), (1, 1)])
def test_milestone_checkpoint_hook_rejects_invalid_milestones(milestones):
    from projects.OVCapFlow.ov_capflow.milestone_checkpoint_hook import (
        MilestoneCheckpointHook, )

    with pytest.raises(ValueError, match='strictly increasing positive'):
        MilestoneCheckpointHook(milestones=milestones)


def test_milestone_checkpoint_hook_is_registered():
    from mmrotate.registry import HOOKS
    from projects.OVCapFlow.ov_capflow.milestone_checkpoint_hook import (
        MilestoneCheckpointHook, )

    assert HOOKS.get('MilestoneCheckpointHook') is MilestoneCheckpointHook

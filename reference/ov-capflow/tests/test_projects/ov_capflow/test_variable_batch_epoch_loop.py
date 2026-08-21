from projects.OVCapFlow.ov_capflow.variable_batch_epoch_loop import (
    collect_epoch_lengths,
)


class _VariableLengthSampler:

    def __init__(self):
        self.epoch = 7

    def set_epoch(self, epoch):
        self.epoch = epoch


class _VariableLengthDataloader:

    def __init__(self, epoch_lengths):
        self.sampler = _VariableLengthSampler()
        self.epoch_lengths = tuple(epoch_lengths)

    def __len__(self):
        return self.epoch_lengths[self.sampler.epoch]


def test_collect_epoch_lengths_uses_each_sampler_epoch_and_restores_state():
    dataloader = _VariableLengthDataloader([402, 403, 401, 404])

    lengths = collect_epoch_lengths(dataloader, max_epochs=4)

    assert lengths == (402, 403, 401, 404)
    assert sum(lengths) == 1610
    assert dataloader.sampler.epoch == 7


class _FixedLengthDataloader:

    def __len__(self):
        return 19


def test_collect_epoch_lengths_falls_back_for_fixed_sampler():
    assert collect_epoch_lengths(
        _FixedLengthDataloader(), max_epochs=3) == (19, 19, 19)

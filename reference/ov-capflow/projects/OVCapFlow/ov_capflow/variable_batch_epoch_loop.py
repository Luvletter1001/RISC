from typing import Optional, Tuple

from torch.utils.data import DataLoader

from mmengine.runner.loops import EpochBasedTrainLoop
from mmrotate.registry import LOOPS


def _find_epoch_sampler(dataloader: DataLoader):
    sampler = getattr(dataloader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        return sampler

    batch_sampler = getattr(dataloader, 'batch_sampler', None)
    sampler = getattr(batch_sampler, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        return sampler
    return None


def collect_epoch_lengths(
        dataloader: DataLoader, max_epochs: int) -> Tuple[int, ...]:
    """Collect exact dataloader lengths for epoch-dependent batch plans."""
    if max_epochs < 1:
        raise ValueError('max_epochs must be positive')

    sampler = _find_epoch_sampler(dataloader)
    if sampler is None:
        return (len(dataloader),) * max_epochs

    original_epoch = int(getattr(sampler, 'epoch', 0))
    try:
        lengths = []
        for epoch in range(max_epochs):
            sampler.set_epoch(epoch)
            lengths.append(len(dataloader))
    finally:
        sampler.set_epoch(original_epoch)
    return tuple(lengths)


@LOOPS.register_module()
class VariableBatchEpochBasedTrainLoop(EpochBasedTrainLoop):
    """Epoch loop with an exact iteration budget for variable batch plans.

    ``EpochBasedTrainLoop`` estimates ``max_iters`` as the first epoch length
    multiplied by the epoch count. That estimate is invalid when a batch
    sampler rebuilds a different query-budget plan after ``set_epoch``.
    The optimizer wrapper uses ``max_iters`` to scale the final gradient
    accumulation group, so an underestimate can eventually produce a zero
    loss factor. This loop enumerates the deterministic epoch lengths once
    and supplies their exact sum instead.
    """

    def __init__(
            self,
            runner,
            dataloader,
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[Tuple[Tuple[int, int], ...]] = None
    ) -> None:
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            max_epochs=max_epochs,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=dynamic_intervals)
        self._epoch_lengths = collect_epoch_lengths(
            self.dataloader, self._max_epochs)
        self._max_iters = sum(self._epoch_lengths)

    @property
    def epoch_lengths(self) -> Tuple[int, ...]:
        return self._epoch_lengths

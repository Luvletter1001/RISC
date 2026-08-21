from pathlib import Path
from typing import Mapping, Sequence, Union


def rebase_checkpoint_iteration_state(
        checkpoint: Mapping,
        epoch_lengths: Sequence[int],
        output_checkpoint: Union[str, Path]):
    """Rebase iteration counters after changing DDP world size.

    The function only copies the small checkpoint metadata containers. Model
    and optimizer states remain shared with the loaded checkpoint so a large
    checkpoint is not duplicated in host memory.
    """
    meta = dict(checkpoint['meta'])
    message_hub = dict(checkpoint['message_hub'])
    runtime_info = dict(message_hub['runtime_info'])
    epoch = int(meta['epoch'])
    old_iter = int(meta['iter'])

    lengths = tuple(int(length) for length in epoch_lengths)
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError('epoch_lengths must contain positive lengths')
    if epoch < 1 or epoch > len(lengths):
        raise ValueError('checkpoint epoch is outside epoch_lengths')
    if (int(runtime_info['epoch']) != epoch - 1
            or int(runtime_info['iter']) != old_iter - 1):
        raise ValueError('checkpoint is not at an epoch boundary')

    new_iter = sum(lengths[:epoch])
    new_max_iters = sum(lengths)
    schedulers = []
    for scheduler in checkpoint.get('param_schedulers', []):
        scheduler = dict(scheduler)
        if not scheduler.get('by_epoch', False):
            if int(scheduler.get('_global_step', old_iter)) < int(
                    scheduler.get('end', 0)):
                raise ValueError(
                    'cannot rebase an active iteration scheduler')
            scheduler['_global_step'] = new_iter
        schedulers.append(scheduler)

    rebased = dict(checkpoint)
    meta['iter'] = new_iter
    runtime_info.update(
        iter=new_iter - 1,
        max_epochs=len(lengths),
        max_iters=new_max_iters,
        last_ckpt=str(output_checkpoint))
    message_hub['runtime_info'] = runtime_info
    rebased['meta'] = meta
    rebased['message_hub'] = message_hub
    rebased['param_schedulers'] = schedulers
    return rebased

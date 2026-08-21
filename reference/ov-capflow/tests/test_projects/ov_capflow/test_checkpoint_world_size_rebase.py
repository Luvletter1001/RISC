import copy

import pytest

from projects.OVCapFlow.ov_capflow.checkpoint_world_size_rebase import (
    rebase_checkpoint_iteration_state,
)


def _checkpoint():
    return {
        'meta': {
            'epoch': 6,
            'iter': 70992,
            'seed': 20260716,
        },
        'message_hub': {
            'log_scalars': {'train/loss': object()},
            'runtime_info': {
                'epoch': 5,
                'iter': 70991,
                'max_epochs': 24,
                'max_iters': 283968,
                'last_ckpt': '/old/epoch_6.pth',
            },
            'resumed_keys': {'iter': True, 'max_iters': True},
        },
        'param_schedulers': [
            {
                'by_epoch': False,
                'begin': 0,
                'end': 500,
                '_global_step': 70992,
                'last_step': 499,
            },
            {
                'by_epoch': True,
                'begin': 0,
                'end': 24,
                '_global_step': 6,
                'last_step': 6,
            },
        ],
        'state_dict': {'weight': object()},
        'optimizer': {'state': object()},
    }


def test_rebase_epoch6_two_gpu_checkpoint_for_four_gpu_epoch_lengths():
    checkpoint = _checkpoint()
    original = copy.deepcopy(checkpoint)
    epoch_lengths = (5916,) * 24

    rebased = rebase_checkpoint_iteration_state(
        checkpoint,
        epoch_lengths=epoch_lengths,
        output_checkpoint='/new/epoch_6_world4.pth')

    assert rebased['meta']['epoch'] == 6
    assert rebased['meta']['iter'] == 35496
    assert rebased['message_hub']['runtime_info']['epoch'] == 5
    assert rebased['message_hub']['runtime_info']['iter'] == 35495
    assert rebased['message_hub']['runtime_info']['max_epochs'] == 24
    assert rebased['message_hub']['runtime_info']['max_iters'] == 142000 - 16
    assert rebased['message_hub']['runtime_info']['last_ckpt'] == (
        '/new/epoch_6_world4.pth')
    assert rebased['param_schedulers'][0]['_global_step'] == 35496
    assert rebased['param_schedulers'][0]['last_step'] == 499
    assert rebased['param_schedulers'][1] == original['param_schedulers'][1]
    assert rebased['state_dict'] is checkpoint['state_dict']
    assert rebased['optimizer'] is checkpoint['optimizer']
    assert checkpoint['meta']['iter'] == 70992


def test_rebase_rejects_checkpoint_that_is_not_at_epoch_boundary():
    checkpoint = _checkpoint()
    checkpoint['message_hub']['runtime_info']['iter'] = 70990

    with pytest.raises(ValueError, match='epoch boundary'):
        rebase_checkpoint_iteration_state(
            checkpoint,
            epoch_lengths=(5916,) * 24,
            output_checkpoint='/new/epoch_6_world4.pth')


def test_rebase_rejects_active_iteration_scheduler():
    checkpoint = _checkpoint()
    checkpoint['param_schedulers'][0]['end'] = 100000

    with pytest.raises(ValueError, match='active iteration scheduler'):
        rebase_checkpoint_iteration_state(
            checkpoint,
            epoch_lengths=(5916,) * 24,
            output_checkpoint='/new/epoch_6_world4.pth')

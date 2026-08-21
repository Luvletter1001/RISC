from pathlib import Path

import pytest
import torch

from projects.OVCapFlow.tools.average_checkpoints import average_checkpoints


def _save(path: Path, weight, counter=7):
    torch.save(
        dict(
            meta=dict(epoch=12),
            state_dict=dict(
                weight=torch.tensor(weight, dtype=torch.float32),
                counter=torch.tensor(counter, dtype=torch.int64)),
            optimizer=dict(should_not_be_copied=True)),
        path)


def test_average_checkpoints_averages_only_floating_state(tmp_path):
    sources = [tmp_path / f'epoch_{index}.pth' for index in range(3)]
    _save(sources[0], [1.0, 3.0])
    _save(sources[1], [3.0, 5.0])
    _save(sources[2], [5.0, 7.0])
    output = tmp_path / 'average.pth'

    report = average_checkpoints(sources, output)
    checkpoint = torch.load(output, map_location='cpu')

    assert torch.equal(
        checkpoint['state_dict']['weight'], torch.tensor([3.0, 5.0]))
    assert checkpoint['state_dict']['counter'].item() == 7
    assert 'optimizer' not in checkpoint
    assert report['floating_tensor_count'] == 1
    assert report['non_floating_tensor_count'] == 1
    assert len(checkpoint['meta']['checkpoint_average']['sources']) == 3


def test_average_checkpoints_rejects_non_floating_mismatch(tmp_path):
    first = tmp_path / 'first.pth'
    second = tmp_path / 'second.pth'
    _save(first, [1.0], counter=7)
    _save(second, [3.0], counter=8)

    with pytest.raises(ValueError, match='non-floating tensor differs'):
        average_checkpoints([first, second], tmp_path / 'average.pth')


def test_average_checkpoints_rejects_duplicate_input(tmp_path):
    source = tmp_path / 'source.pth'
    _save(source, [1.0])

    with pytest.raises(ValueError, match='must be distinct'):
        average_checkpoints([source, source], tmp_path / 'average.pth')

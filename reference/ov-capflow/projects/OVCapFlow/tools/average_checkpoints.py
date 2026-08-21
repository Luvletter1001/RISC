"""Create a deterministic uniform average of model checkpoints.

Only floating-point tensors are averaged. Non-floating tensors are required to
be exactly equal across every input, which prevents silently combining
incompatible model states. Optimizer and scheduler state are intentionally not
copied because the output is an evaluation/promotion artifact, not a resume
checkpoint.
"""

import argparse
import gc
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> Mapping[str, Tensor]:
    checkpoint = torch.load(str(path), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'{path}: checkpoint must be a mapping')
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise TypeError(f'{path}: checkpoint must contain state_dict')
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError(f'{path}: state_dict must map strings to tensors')
    return state


def average_checkpoints(inputs: Sequence[Path], output: Path) -> Dict:
    inputs = [Path(path).expanduser().resolve() for path in inputs]
    output = Path(output).expanduser().resolve()
    if len(inputs) < 2:
        raise ValueError('at least two input checkpoints are required')
    if len(set(inputs)) != len(inputs):
        raise ValueError('input checkpoints must be distinct')
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing checkpoints: ' + ', '.join(missing))

    count = len(inputs)
    first_checkpoint = torch.load(str(inputs[0]), map_location='cpu')
    if not isinstance(first_checkpoint, Mapping):
        raise TypeError(f'{inputs[0]}: checkpoint must be a mapping')
    first_state = first_checkpoint.get('state_dict')
    if not isinstance(first_state, Mapping):
        raise TypeError(f'{inputs[0]}: checkpoint must contain state_dict')

    averaged = OrderedDict()
    for key, value in first_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError(
                f'{inputs[0]}: state_dict must map strings to tensors')
        averaged[key] = value.clone()
        if value.is_floating_point():
            averaged[key].div_(count)

    reference_keys = tuple(averaged.keys())
    del first_state
    for path in inputs[1:]:
        state = _load_state(path)
        if tuple(state.keys()) != reference_keys:
            raise ValueError(f'{path}: state_dict keys or order differ')
        for key, value in state.items():
            target = averaged[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f'{path}: incompatible tensor {key}')
            if value.is_floating_point():
                target.add_(value, alpha=1.0 / count)
            elif not torch.equal(target, value):
                raise ValueError(
                    f'{path}: non-floating tensor differs: {key}')
        del state
        gc.collect()

    source_records: List[Dict[str, str]] = [
        dict(path=str(path), sha256=sha256_file(path)) for path in inputs
    ]
    meta = dict(first_checkpoint.get('meta', {}))
    meta['checkpoint_average'] = dict(
        method='uniform_arithmetic_mean',
        floating_tensor_count=sum(
            tensor.is_floating_point() for tensor in averaged.values()),
        non_floating_tensor_count=sum(
            not tensor.is_floating_point() for tensor in averaged.values()),
        sources=source_records)
    artifact = dict(meta=meta, state_dict=averaged)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    torch.save(artifact, str(temporary))
    temporary.replace(output)
    return dict(
        output=str(output),
        output_sha256=sha256_file(output),
        sources=source_records,
        tensor_count=len(averaged),
        floating_tensor_count=meta['checkpoint_average'][
            'floating_tensor_count'],
        non_floating_tensor_count=meta['checkpoint_average'][
            'non_floating_tensor_count'])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(average_checkpoints(args.input, args.output))


if __name__ == '__main__':
    main()

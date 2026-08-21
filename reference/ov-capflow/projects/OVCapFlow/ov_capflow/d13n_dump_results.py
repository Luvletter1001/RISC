"""Distributed result dumping with immutable atomic publication."""

import logging
import types
from typing import Any, BinaryIO, Optional, Sequence

import mmengine.evaluator.metric as metric_module
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.registry import METRICS
from mmengine.structures import BaseDataElement

from .no_replace import _collision_paths, publish_stream_noreplace


_EVALUATE_ENVELOPE_SCHEMA = 'd13n-dump-evaluate-v1'


class D13NDumpCollisionError(FileExistsError):
    """A rank-zero dump collision reconstructed identically on every rank."""

    def __init__(self,
                 out_file_path: str,
                 original_errno: Optional[int],
                 original_filename: Optional[str],
                 original_filename2: Optional[str],
                 original_message: str) -> None:
        super().__init__(original_message)
        self.out_file_path = str(out_file_path)
        self.original_errno = original_errno
        self.original_filename = original_filename
        self.original_filename2 = original_filename2
        self.original_message = original_message
        # Preserve the standard OSError fields as well as the explicit wire
        # fields above.  Calling OSError with a synthetic argument tuple would
        # alter the original message, especially for native link(2) failures.
        self.errno = original_errno
        self.filename = original_filename
        self.filename2 = original_filename2

    @classmethod
    def from_error(cls, out_file_path: str,
                   error: FileExistsError) -> 'D13NDumpCollisionError':
        if isinstance(error, cls):
            return cls(
                out_file_path=str(out_file_path),
                original_errno=error.original_errno,
                original_filename=error.original_filename,
                original_filename2=error.original_filename2,
                original_message=error.original_message)
        return cls(
            out_file_path=str(out_file_path),
            original_errno=getattr(error, 'errno', None),
            original_filename=getattr(error, 'filename', None),
            original_filename2=getattr(error, 'filename2', None),
            original_message=str(error))

    def __str__(self) -> str:
        return self.original_message


class _DumpWriterFileExistsError(RuntimeError):
    """Keep a writer-local FileExistsError out of publisher classification."""

    def __init__(self, original: FileExistsError) -> None:
        super().__init__(str(original))
        self.original = original


def _success_envelope(metrics: dict) -> dict:
    return {
        'schema': _EVALUATE_ENVELOPE_SCHEMA,
        'status': 'ok',
        'metrics': metrics,
    }


def _error_envelope(out_file_path: str, error: Exception) -> dict:
    if isinstance(error, D13NDumpCollisionError):
        collision = D13NDumpCollisionError.from_error(out_file_path, error)
        payload = {
            'kind': 'collision',
            'out_file_path': collision.out_file_path,
            'original_errno': collision.original_errno,
            'original_filename': collision.original_filename,
            'original_filename2': collision.original_filename2,
            'original_message': collision.original_message,
        }
    else:
        payload = {
            'kind': 'runtime',
            'error_type': type(error).__name__,
            'message': str(error),
        }
    return {
        'schema': _EVALUATE_ENVELOPE_SCHEMA,
        'status': 'error',
        'error': payload,
    }


def _unwrap_evaluate_envelope(envelope: Any) -> dict:
    if not isinstance(envelope, dict):
        raise RuntimeError('invalid D13-N dump evaluate envelope')
    if envelope.get('schema') != _EVALUATE_ENVELOPE_SCHEMA:
        raise RuntimeError('invalid D13-N dump evaluate envelope schema')
    status = envelope.get('status')
    if status == 'ok':
        metrics = envelope.get('metrics')
        if not isinstance(metrics, dict):
            raise RuntimeError('invalid D13-N dump evaluate metrics')
        return metrics
    if status != 'error' or not isinstance(envelope.get('error'), dict):
        raise RuntimeError('invalid D13-N dump evaluate status')
    error = envelope['error']
    if error.get('kind') == 'collision':
        if (type(error.get('out_file_path')) is not str
                or (error.get('original_errno') is not None
                    and type(error.get('original_errno')) is not int)
                or (error.get('original_filename') is not None
                    and type(error.get('original_filename')) is not str)
                or (error.get('original_filename2') is not None
                    and type(error.get('original_filename2')) is not str)
                or type(error.get('original_message')) is not str):
            raise RuntimeError('invalid D13-N dump collision envelope')
        raise D13NDumpCollisionError(
            out_file_path=error.get('out_file_path'),
            original_errno=error.get('original_errno'),
            original_filename=error.get('original_filename'),
            original_filename2=error.get('original_filename2'),
            original_message=error.get('original_message'))
    if error.get('kind') == 'runtime':
        if (type(error.get('error_type')) is not str
                or type(error.get('message')) is not str):
            raise RuntimeError('invalid D13-N dump runtime envelope')
        raise RuntimeError('{}: {}'.format(
            error.get('error_type'), error.get('message')))
    raise RuntimeError('invalid D13-N dump evaluate error')


def cpu_prediction_tree(value: Any) -> Any:
    """Recursively move prediction tensors to CPU without reordering them."""
    return _cpu_prediction_tree(value, {})


def _shallow_clone_data_element(value: BaseDataElement) -> BaseDataElement:
    """Clone outer BaseDataElement state without running construction hooks."""
    converted = object.__new__(type(value))
    if hasattr(value, '__dict__'):
        converted.__dict__.update(value.__dict__)
    for cls in type(value).__mro__:
        for descriptor in cls.__dict__.values():
            if not isinstance(descriptor, types.MemberDescriptorType):
                continue
            try:
                slot_value = descriptor.__get__(value, type(value))
            except AttributeError:
                continue
            descriptor.__set__(converted, slot_value)
    return converted


def _cpu_prediction_tree(value: Any, memo: dict) -> Any:
    """Memoized implementation for structured and repeated predictions."""
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        converted = value.to('cpu')
        memo[value_id] = converted
        return converted
    if isinstance(value, BaseDataElement):
        converted = _shallow_clone_data_element(value)
        object.__setattr__(
            converted, '_metainfo_fields', value._metainfo_fields.copy())
        object.__setattr__(
            converted, '_data_fields', value._data_fields.copy())
        memo[value_id] = converted
        for name, field_value in value.metainfo_items():
            converted.set_field(
                _cpu_prediction_tree(field_value, memo),
                name,
                field_type='metainfo')
        for name, field_value in value.items():
            setattr(converted, name, _cpu_prediction_tree(field_value, memo))
        return converted
    if isinstance(value, list):
        converted = []
        memo[value_id] = converted
        converted.extend(_cpu_prediction_tree(item, memo) for item in value)
        return converted
    if isinstance(value, tuple):
        items = tuple(_cpu_prediction_tree(item, memo) for item in value)
        if hasattr(type(value), '_fields'):
            converted = type(value)(*items)
        else:
            converted = items
        memo[value_id] = converted
        return converted
    if isinstance(value, dict):
        converted = {}
        memo[value_id] = converted
        converted.update((key, _cpu_prediction_tree(item, memo))
                         for key, item in value.items())
        return converted
    return value


def dump_results_to_stream(results: list, stream: BinaryIO) -> None:
    """Write one MMEngine-compatible pickle directly to ``stream``."""
    dump(results, stream, file_format='pkl')


@METRICS.register_module()
class D13NNoReplaceDumpResults(BaseMetric):
    """Gather ordered predictions and publish one immutable pickle dump."""

    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')
        self.out_file_path = out_file_path

    def process(self, data_batch: Any,
                predictions: Sequence[dict]) -> None:
        """Move predictions to CPU before distributed collection."""
        self.results.extend(cpu_prediction_tree(predictions))

    def evaluate(self, size: int) -> dict:
        """Collect and dump without MMEngine's unsafe second ``_to_cpu``."""
        if len(self.results) == 0:
            metric_module.print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = metric_module.collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = metric_module.collect_results(
                self.results, size, self.collect_device)

        envelope = [None]
        if metric_module.is_main_process():
            try:
                results = cpu_prediction_tree(results)
                computed = self.compute_metrics(results)
                if self.prefix:
                    computed = {
                        '/'.join((self.prefix, key)): value
                        for key, value in computed.items()
                    }
                envelope[0] = _success_envelope(computed)
            except Exception as error:
                envelope[0] = _error_envelope(self.out_file_path, error)

        try:
            metric_module.broadcast_object_list(envelope)
        finally:
            self.results.clear()
        return _unwrap_evaluate_envelope(envelope[0])

    def compute_metrics(self, results: list) -> dict:
        """Publish the complete ordered result list without replacement."""
        def write_results(stream: BinaryIO) -> None:
            try:
                dump_results_to_stream(results, stream)
            except FileExistsError as error:
                raise _DumpWriterFileExistsError(error) from error

        try:
            publish_stream_noreplace(self.out_file_path, write_results)
        except _DumpWriterFileExistsError as error:
            raise error.original
        except FileExistsError as error:
            if _collision_paths(self.out_file_path):
                raise D13NDumpCollisionError.from_error(
                    self.out_file_path, error) from error
            raise
        print_log(
            f'Results has been saved to {self.out_file_path}.',
            logger='current')
        return {}


__all__ = [
    'D13NDumpCollisionError',
    'D13NNoReplaceDumpResults',
    'cpu_prediction_tree',
    'dump_results_to_stream',
]

import errno
import io
import logging
import multiprocessing
import os
import threading
from collections import OrderedDict
from pathlib import Path

import mmengine
import pytest
import torch
from mmengine.evaluator import BaseMetric
from mmengine.structures import BaseDataElement


class _DeviceMoveProbeTensor(torch.Tensor):
    moves = []

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(
            cls, value, require_grad=False)

    def to(self, *args, **kwargs):
        type(self).moves.append((args, kwargs, tuple(self.tolist())))
        return torch.tensor(
            self.tolist(), dtype=self.dtype, device='cpu')


class _ForbiddenNonTensorTo:

    def __init__(self):
        self.calls = 0

    def to(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError('non-tensor .to() must never be called')


class _RequiredSlotDataElement(BaseDataElement):
    __slots__ = ('required_token', 'unregistered_state', 'slot_metainfo',
                 'slot_data')

    def __init__(self, required_token):
        super().__init__()
        object.__setattr__(self, 'required_token', required_token)
        object.__setattr__(self, 'unregistered_state', {'owner': required_token})


def _api():
    from projects.OVCapFlow.ov_capflow import d13n_dump_results

    return d13n_dump_results


def _gloo_dump_collision_worker(
        rank, world_size, init_file, target, collision_kind, queue):
    import torch.distributed as dist
    import mmengine.evaluator.metric as metric_module
    from projects.OVCapFlow.ov_capflow import d13n_dump_results as api
    from projects.OVCapFlow.ov_capflow import no_replace

    dist.init_process_group(
        backend='gloo',
        init_method='file://' + init_file,
        rank=rank,
        world_size=world_size)
    metric_module.collect_results = (
        lambda results, size, device, tmpdir=None:
        [{'rank': index} for index in range(world_size)]
        if rank == 0 else None)
    metric = api.D13NNoReplaceDumpResults(target)
    metric.results.append({'local_rank': rank})
    if collision_kind == 'native-link' and rank == 0:
        real_link = no_replace.os.link

        def racing_link(source, destination):
            Path(destination).write_bytes(b'native-race-winner')
            return real_link(source, destination)

        no_replace.os.link = racing_link
    try:
        metric.evaluate(size=world_size)
        queue.put({'rank': rank, 'unexpected_success': True})
    except Exception as error:
        queue.put({
            'rank': rank,
            'type': type(error).__name__,
            'out_file_path': getattr(error, 'out_file_path', None),
            'original_errno': getattr(error, 'original_errno', None),
            'original_filename': getattr(error, 'original_filename', None),
            'original_filename2': getattr(error, 'original_filename2', None),
            'original_message': getattr(error, 'original_message', None),
            'cleared': metric.results == [],
        })
    finally:
        dist.destroy_process_group()


def _run_gloo_dump_collision(tmp_path, collision_kind, world_size):
    target = tmp_path / 'distributed, collision.pkl'
    if collision_kind == 'preflight':
        target.write_bytes(b'preflight-owner')
    init_file = tmp_path / 'gloo-init'
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    processes = [
        context.Process(
            target=_gloo_dump_collision_worker,
            args=(rank, world_size, str(init_file), str(target), collision_kind,
                  queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert hung == []
    assert [process.exitcode for process in processes] == [0] * world_size
    records = sorted((queue.get(timeout=5) for _ in range(world_size)),
                     key=lambda row: row['rank'])
    return target, records


def test_cpu_prediction_tree_preserves_nested_values_and_order():
    api = _api()
    source = OrderedDict([
        ('first', torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)),
        ('second', [
            'keep-me',
            OrderedDict([
                ('score', torch.tensor([0.25, 0.75], dtype=torch.float64)),
                ('flag', True),
            ]),
        ]),
    ])

    converted = api.cpu_prediction_tree(source)

    assert list(converted) == ['first', 'second']
    assert list(converted['second'][1]) == ['score', 'flag']
    assert converted['second'][0] == 'keep-me'
    assert converted['second'][1]['flag'] is True
    for actual, expected in [
            (converted['first'], source['first']),
            (converted['second'][1]['score'], source['second'][1]['score'])]:
        assert actual.device.type == 'cpu'
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


def test_cpu_prediction_tree_rebuilds_data_elements_and_all_nested_fields():
    api = _api()
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    _DeviceMoveProbeTensor.moves = []

    def probe(*values):
        return _DeviceMoveProbeTensor(
            torch.tensor(values, dtype=torch.float64))

    instances = InstanceData()
    instances.scores = probe(0.25, 0.75)
    instances.labels = torch.tensor([3, 7], dtype=torch.int64)
    instances.nested = [
        {'trace': probe(10.0)},
        {'trace': probe(20.0)},
    ]
    sample = DetDataSample()
    sample.set_field(
        {'levels': [probe(30.0), {'tuple': (probe(40.0), 'meta')}]},
        'nested_metainfo',
        field_type='metainfo')
    sample.pred_instances = instances
    sample.direct_tensor = probe(50.0)
    sample.nested_data = [
        probe(60.0),
        OrderedDict([('inside', (probe(70.0), 'data'))]),
    ]

    converted = api.cpu_prediction_tree(sample)

    assert type(converted) is DetDataSample
    assert type(converted.pred_instances) is InstanceData
    assert tuple(converted.__dict__) == tuple(sample.__dict__)
    assert tuple(converted.pred_instances.__dict__) == tuple(instances.__dict__)
    assert converted._metainfo_fields == sample._metainfo_fields
    assert converted._data_fields == sample._data_fields
    assert converted.pred_instances._data_fields == instances._data_fields
    assert list(converted.metainfo_keys()) == list(sample.metainfo_keys())
    assert list(converted.keys()) == list(sample.keys())
    assert list(converted.pred_instances.keys()) == list(instances.keys())
    assert converted.pred_instances.labels.dtype == torch.int64
    assert converted.pred_instances.labels.device.type == 'cpu'
    assert torch.equal(converted.pred_instances.labels,
                       torch.tensor([3, 7], dtype=torch.int64))
    assert converted.nested_metainfo['levels'][0].device.type == 'cpu'
    assert converted.nested_metainfo['levels'][1]['tuple'][0].device.type == 'cpu'
    assert converted.pred_instances.nested[0]['trace'].device.type == 'cpu'
    assert converted.pred_instances.nested[1]['trace'].device.type == 'cpu'
    assert converted.direct_tensor.device.type == 'cpu'
    assert converted.nested_data[0].device.type == 'cpu'
    assert converted.nested_data[1]['inside'][0].device.type == 'cpu'
    assert converted.nested_metainfo['levels'][1]['tuple'][1] == 'meta'
    assert converted.nested_data[1]['inside'][1] == 'data'
    for actual, expected in [
            (converted.pred_instances.scores, (0.25, 0.75)),
            (converted.pred_instances.nested[0]['trace'], (10.0, )),
            (converted.pred_instances.nested[1]['trace'], (20.0, )),
            (converted.nested_metainfo['levels'][0], (30.0, )),
            (converted.nested_metainfo['levels'][1]['tuple'][0], (40.0, )),
            (converted.direct_tensor, (50.0, )),
            (converted.nested_data[0], (60.0, )),
            (converted.nested_data[1]['inside'][0], (70.0, )),
    ]:
        assert actual.dtype == torch.float64
        assert actual.shape == (len(expected), )
        assert torch.equal(actual, torch.tensor(expected, dtype=torch.float64))
    assert sorted(move[2] for move in _DeviceMoveProbeTensor.moves) == [
        (0.25, 0.75),
        (10.0, ),
        (20.0, ),
        (30.0, ),
        (40.0, ),
        (50.0, ),
        (60.0, ),
        (70.0, ),
    ]
    assert all(move[0] == ('cpu', ) and move[1] == {}
               for move in _DeviceMoveProbeTensor.moves)


def test_cpu_prediction_tree_never_calls_to_on_non_tensor_fields():
    api = _api()
    from mmdet.structures import DetDataSample

    opaque = _ForbiddenNonTensorTo()
    sample = DetDataSample()
    sample.set_field(
        {'opaque': opaque}, 'opaque_metainfo', field_type='metainfo')
    sample.opaque = opaque
    sample.nested = [opaque, {'same': opaque}]

    converted = api.cpu_prediction_tree(sample)

    assert opaque.calls == 0
    assert converted.opaque_metainfo['opaque'] is opaque
    assert converted.opaque is opaque
    assert converted.nested[0] is opaque
    assert converted.nested[1]['same'] is opaque


def test_cpu_prediction_tree_preserves_slots_required_init_and_outer_state():
    api = _api()
    _DeviceMoveProbeTensor.moves = []
    required_token = object()
    source = _RequiredSlotDataElement(required_token)
    source.set_field(
        {'nested': [_DeviceMoveProbeTensor(torch.tensor([11.0]))]},
        'slot_metainfo',
        field_type='metainfo')
    source.set_field(
        [_DeviceMoveProbeTensor(torch.tensor([21.0])), 'slot-value'],
        'slot_data',
        field_type='data')

    converted = api.cpu_prediction_tree(source)

    assert type(converted) is _RequiredSlotDataElement
    assert converted is not source
    assert converted.required_token is required_token
    assert converted.unregistered_state is source.unregistered_state
    assert converted._metainfo_fields == source._metainfo_fields
    assert converted._data_fields == source._data_fields
    assert list(converted.metainfo_keys()) == list(source.metainfo_keys())
    assert list(converted.keys()) == list(source.keys())
    assert converted.slot_metainfo['nested'][0].device.type == 'cpu'
    assert converted.slot_metainfo['nested'][0].item() == 11.0
    assert converted.slot_data[0].device.type == 'cpu'
    assert converted.slot_data[0].item() == 21.0
    assert converted.slot_data[1] == 'slot-value'
    assert sorted(move[2] for move in _DeviceMoveProbeTensor.moves) == [
        (11.0, ), (21.0, )
    ]


def test_dump_results_to_stream_is_one_mmengine_readable_pickle():
    api = _api()
    results = [
        {'img_id': 'a', 'scores': torch.tensor([0.5], dtype=torch.float32)},
        {'img_id': 'b', 'labels': [2, 1]},
    ]
    stream = io.BytesIO()

    api.dump_results_to_stream(results, stream)

    stream.seek(0)
    restored = mmengine.load(stream, file_format='pkl')
    assert restored[0]['img_id'] == 'a'
    assert torch.equal(restored[0]['scores'], results[0]['scores'])
    assert restored[1] == results[1]
    assert stream.read() == b''


def test_metric_extends_base_metric_validates_suffix_and_processes_to_cpu(
        tmp_path):
    api = _api()
    target = tmp_path / 'predictions.pkl'

    metric = api.D13NNoReplaceDumpResults(
        str(target), collect_device='cpu', collect_dir=str(tmp_path / 'parts'))

    assert isinstance(metric, BaseMetric)
    assert metric.out_file_path == str(target)
    assert metric.collect_device == 'cpu'
    assert metric.collect_dir == str(tmp_path / 'parts')
    metric.process({}, [{
        'img_id': 'x',
        'nested': [torch.tensor([1.0], dtype=torch.float64), 'unchanged'],
    }])
    assert metric.results[0]['nested'][0].device.type == 'cpu'
    assert metric.results[0]['nested'][0].dtype == torch.float64
    assert metric.results[0]['nested'][1] == 'unchanged'

    with pytest.raises(ValueError, match='pkl'):
        api.D13NNoReplaceDumpResults(str(tmp_path / 'predictions.json'))


def test_evaluate_uses_base_metric_ordered_collection_and_publishes_once(
        tmp_path, monkeypatch):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    target = tmp_path / 'distributed.pkl'
    collect_dir = tmp_path / 'collect'
    metric = api.D13NNoReplaceDumpResults(
        str(target), collect_device='cpu', collect_dir=str(collect_dir))
    metric.results.extend([{'local': 0}, {'local': 1}])
    ordered = [
        {'rank': 0, 'row': 0},
        {'rank': 1, 'row': 0},
        {'rank': 0, 'row': 1},
        {'rank': 1, 'row': 1},
    ]
    calls = []

    def collect_results(results, size, device, tmpdir=None):
        calls.append((list(results), size, device, tmpdir))
        return ordered

    monkeypatch.setattr(metric_module, 'collect_results', collect_results)
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: True)
    monkeypatch.setattr(metric_module, 'broadcast_object_list', lambda value: None)

    assert metric.evaluate(size=4) == {}

    assert calls == [([{'local': 0}, {'local': 1}], 4, 'cpu',
                      str(collect_dir))]
    assert mmengine.load(str(target)) == ordered
    assert metric.results == []


def test_real_process_evaluate_publishes_structured_predictions_without_to(
        tmp_path):
    api = _api()
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    target = tmp_path / 'structured.pkl'
    opaque = _ForbiddenNonTensorTo()
    sample = DetDataSample()
    sample.set_field(
        {'nested': [torch.tensor([9.0], dtype=torch.float64)]},
        'trace_metainfo',
        field_type='metainfo')
    sample.pred_instances = InstanceData(
        scores=torch.tensor([0.9], dtype=torch.float32),
        labels=torch.tensor([4], dtype=torch.int64))
    sample.nested = [{'tensor': torch.tensor([[3, 5]], dtype=torch.int32)}]
    sample.opaque = opaque
    metric = api.D13NNoReplaceDumpResults(str(target))

    metric.process({}, [sample])
    assert metric.evaluate(size=1) == {}

    assert opaque.calls == 0
    restored = mmengine.load(str(target))
    assert len(restored) == 1
    restored_sample = restored[0]
    assert type(restored_sample) is DetDataSample
    assert type(restored_sample.pred_instances) is InstanceData
    assert restored_sample._metainfo_fields == sample._metainfo_fields
    assert restored_sample._data_fields == sample._data_fields
    assert torch.equal(
        restored_sample.trace_metainfo['nested'][0],
        torch.tensor([9.0], dtype=torch.float64))
    assert torch.equal(
        restored_sample.pred_instances.scores,
        torch.tensor([0.9], dtype=torch.float32))
    assert torch.equal(
        restored_sample.pred_instances.labels,
        torch.tensor([4], dtype=torch.int64))
    assert torch.equal(
        restored_sample.nested[0]['tensor'],
        torch.tensor([[3, 5]], dtype=torch.int32))
    assert restored_sample.opaque.calls == 0


def test_evaluate_non_main_gpu_path_broadcasts_clears_and_never_publishes(
        tmp_path, monkeypatch):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    target = tmp_path / 'non_main.pkl'
    metric = api.D13NNoReplaceDumpResults(
        str(target), collect_device='gpu')
    collect_calls = []
    warnings = []

    def collect_results(*args, **kwargs):
        collect_calls.append((args, kwargs))
        return None

    def broadcast(metrics):
        assert metrics == [None]
        metrics[0] = {
            'schema': 'd13n-dump-evaluate-v1',
            'status': 'ok',
            'metrics': {'from_main': 7},
        }

    def print_warning(message, logger=None, level=None):
        warnings.append((message, logger, level))

    monkeypatch.setattr(metric_module, 'collect_results', collect_results)
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: False)
    monkeypatch.setattr(metric_module, 'broadcast_object_list', broadcast)
    monkeypatch.setattr(metric_module, 'print_log', print_warning)
    monkeypatch.setattr(
        metric, 'compute_metrics',
        lambda results: pytest.fail('non-main must not compute or publish'))

    assert metric.evaluate(size=3) == {'from_main': 7}

    assert collect_calls == [(([], 3, 'gpu'), {})]
    assert len(warnings) == 1
    assert 'got empty `self.results`' in warnings[0][0]
    assert warnings[0][1:] == ('current', logging.WARNING)
    assert metric.results == []
    assert not target.exists()


def test_evaluate_applies_prefix_before_broadcast_and_clears(
        tmp_path, monkeypatch):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    target = tmp_path / 'prefix.pkl'
    metric = api.D13NNoReplaceDumpResults(str(target))
    metric.prefix = 'd13n'
    metric.results.append({'row': 1})
    broadcast_values = []

    monkeypatch.setattr(
        metric_module, 'collect_results',
        lambda results, size, device, tmpdir=None: [{'row': 1}])
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: True)
    monkeypatch.setattr(
        metric, 'compute_metrics', lambda results: {'records': len(results)})
    monkeypatch.setattr(
        metric_module, 'broadcast_object_list',
        lambda values: broadcast_values.append(list(values)))

    assert metric.evaluate(size=1) == {'d13n/records': 1}

    assert broadcast_values == [[{
        'schema': 'd13n-dump-evaluate-v1',
        'status': 'ok',
        'metrics': {'d13n/records': 1},
    }]]
    assert metric.results == []
    assert not target.exists()


def test_evaluate_broadcasts_runtime_error_and_clears_main_results(
        tmp_path, monkeypatch):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    metric = api.D13NNoReplaceDumpResults(str(tmp_path / 'runtime.pkl'))
    metric.results.append({'row': 1})
    broadcasts = []

    monkeypatch.setattr(
        metric_module, 'collect_results',
        lambda results, size, device, tmpdir=None: [{'row': 1}])
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: True)
    monkeypatch.setattr(
        metric, 'compute_metrics',
        lambda results: (_ for _ in ()).throw(ValueError('late failure')))
    monkeypatch.setattr(
        metric_module, 'broadcast_object_list',
        lambda values: broadcasts.append(list(values)))

    with pytest.raises(RuntimeError, match='ValueError: late failure'):
        metric.evaluate(size=1)

    assert broadcasts == [[{
        'schema': 'd13n-dump-evaluate-v1',
        'status': 'error',
        'error': {
            'kind': 'runtime',
            'error_type': 'ValueError',
            'message': 'late failure',
        },
    }]]
    assert metric.results == []


@pytest.mark.parametrize(
    'origin', ['compute_metrics', 'serialization', 'print_log'])
def test_evaluate_keeps_nonpublication_file_exists_as_runtime_error(
        tmp_path, monkeypatch, origin):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    target = tmp_path / '{}.pkl'.format(origin)
    metric = api.D13NNoReplaceDumpResults(str(target))
    metric.results.append({'row': 1})
    broadcasts = []
    message = 'unrelated {} file exists'.format(origin)

    monkeypatch.setattr(
        metric_module, 'collect_results',
        lambda results, size, device, tmpdir=None: [{'row': 1}])
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: True)
    monkeypatch.setattr(
        metric_module, 'broadcast_object_list',
        lambda values: broadcasts.append(list(values)))
    if origin == 'compute_metrics':
        monkeypatch.setattr(
            metric, 'compute_metrics',
            lambda results: (_ for _ in ()).throw(FileExistsError(message)))
    elif origin == 'serialization':
        monkeypatch.setattr(
            api, 'dump_results_to_stream',
            lambda results, stream:
            (_ for _ in ()).throw(FileExistsError(message)))
    else:
        monkeypatch.setattr(
            api, 'print_log',
            lambda *args, **kwargs:
            (_ for _ in ()).throw(FileExistsError(message)))

    with pytest.raises(
            RuntimeError, match='FileExistsError: {}'.format(message)):
        metric.evaluate(size=1)

    assert broadcasts == [[{
        'schema': 'd13n-dump-evaluate-v1',
        'status': 'error',
        'error': {
            'kind': 'runtime',
            'error_type': 'FileExistsError',
            'message': message,
        },
    }]]
    assert metric.results == []


def test_evaluate_non_main_reconstructs_runtime_error_and_clears(
        tmp_path, monkeypatch):
    api = _api()
    import mmengine.evaluator.metric as metric_module

    metric = api.D13NNoReplaceDumpResults(str(tmp_path / 'runtime.pkl'))
    metric.results.append({'row': 1})

    monkeypatch.setattr(
        metric_module, 'collect_results',
        lambda results, size, device, tmpdir=None: None)
    monkeypatch.setattr(metric_module, 'is_main_process', lambda: False)

    def broadcast(values):
        assert values == [None]
        values[0] = {
            'schema': 'd13n-dump-evaluate-v1',
            'status': 'error',
            'error': {
                'kind': 'runtime',
                'error_type': 'ValueError',
                'message': 'late failure',
            },
        }

    monkeypatch.setattr(metric_module, 'broadcast_object_list', broadcast)

    with pytest.raises(RuntimeError, match='ValueError: late failure'):
        metric.evaluate(size=1)

    assert metric.results == []


@pytest.mark.parametrize('world_size', [2, 5])
@pytest.mark.parametrize('collision_kind', ['preflight', 'native-link'])
def test_gloo_evaluate_converges_dump_collisions_without_hang(
        tmp_path, collision_kind, world_size):
    target, records = _run_gloo_dump_collision(
        tmp_path, collision_kind, world_size)

    assert [record['type'] for record in records] == [
        'D13NDumpCollisionError'] * world_size
    assert [record['out_file_path'] for record in records] == [
        str(target)] * world_size
    assert len({record['original_errno'] for record in records}) == 1
    assert len({record['original_filename'] for record in records}) == 1
    assert len({record['original_filename2'] for record in records}) == 1
    assert len({record['original_message'] for record in records}) == 1
    assert [record['cleared'] for record in records] == [True] * world_size
    if collision_kind == 'preflight':
        assert target.read_bytes() == b'preflight-owner'
        assert records[0]['original_errno'] is None
        assert records[0]['original_filename'] is None
        assert records[0]['original_filename2'] is None
        assert records[0]['original_message'] == f'output collision: {target}'
    else:
        assert target.read_bytes() == b'native-race-winner'
        assert records[0]['original_errno'] == errno.EEXIST
        assert records[0]['original_filename'].startswith(
            str(target) + '.pending.')
        assert records[0]['original_filename2'] == str(target)


@pytest.mark.parametrize('collision_kind', ['final', 'pending'])
def test_metric_refuses_occupied_final_or_pending_and_preserves_bytes(
        tmp_path, collision_kind):
    api = _api()
    target = tmp_path / 'predictions.pkl'
    occupied = target
    if collision_kind == 'pending':
        occupied = tmp_path / 'predictions.pkl.pending.123.0'
    occupied.write_bytes(b'preserve-existing-evidence')
    metric = api.D13NNoReplaceDumpResults(str(target))

    with pytest.raises(FileExistsError, match='output collision'):
        metric.compute_metrics([{'new': 'prediction'}])

    assert occupied.read_bytes() == b'preserve-existing-evidence'
    if collision_kind == 'pending':
        assert not target.exists()


def test_concurrent_metric_publications_keep_one_complete_winner(
        tmp_path, monkeypatch):
    api = _api()
    from projects.OVCapFlow.ov_capflow import no_replace

    target = tmp_path / 'predictions.pkl'
    barrier = threading.Barrier(2)
    real_raise_for_collisions = no_replace._raise_for_collisions

    def synchronized_collision_scan(path):
        real_raise_for_collisions(path)
        barrier.wait(timeout=10)

    monkeypatch.setattr(no_replace, '_raise_for_collisions',
                        synchronized_collision_scan)
    payloads = [
        [{'publisher': 0, 'rows': list(range(2000))}],
        [{'publisher': 1, 'rows': list(range(2000, 4000))}],
    ]
    failures = []

    def publish(payload):
        try:
            api.D13NNoReplaceDumpResults(str(target)).compute_metrics(payload)
        except BaseException as error:
            failures.append(error)

    threads = [
        threading.Thread(target=publish, args=(payload, ))
        for payload in payloads
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert mmengine.load(str(target)) in payloads
    pending = list(tmp_path.glob('predictions.pkl.pending.*'))
    assert len(pending) <= 1
    if pending:
        assert mmengine.load(str(pending[0])) in payloads


def test_reader_observes_no_final_until_complete_pickle_is_written(
        tmp_path, monkeypatch):
    api = _api()
    target = tmp_path / 'predictions.pkl'
    results = [{'img_id': 'image-1', 'rows': list(range(600))}]
    entered = threading.Event()
    release = threading.Event()
    real_dump_results_to_stream = api.dump_results_to_stream

    def blocking_dump(values, stream):
        entered.set()
        assert release.wait(timeout=10)
        real_dump_results_to_stream(values, stream)

    monkeypatch.setattr(api, 'dump_results_to_stream', blocking_dump)
    failure = []

    def publish():
        try:
            api.D13NNoReplaceDumpResults(str(target)).compute_metrics(results)
        except BaseException as error:
            failure.append(error)

    thread = threading.Thread(target=publish)
    thread.start()
    assert entered.wait(timeout=10)
    assert not os.path.lexists(target)
    release.set()
    thread.join(timeout=15)

    assert not thread.is_alive()
    assert failure == []
    assert mmengine.load(str(target)) == results


def test_metric_serializes_directly_into_publisher_stream_without_full_copy(
        tmp_path, monkeypatch):
    api = _api()
    target = tmp_path / 'predictions.pkl'

    class WriteOnlyStream:

        def __init__(self):
            self.payload = bytearray()
            self.write_sizes = []

        def write(self, payload):
            assert type(payload) is bytes
            self.write_sizes.append(len(payload))
            self.payload.extend(payload)
            return len(payload)

    pending_stream = WriteOnlyStream()
    published_paths = []
    dump_streams = []
    real_dump = api.dump

    def publish_stream_noreplace(path, writer):
        published_paths.append(path)
        writer(pending_stream)

    def record_dump(value, stream, **kwargs):
        dump_streams.append(stream)
        return real_dump(value, stream, **kwargs)

    monkeypatch.setattr(api, 'publish_stream_noreplace',
                        publish_stream_noreplace)
    monkeypatch.setattr(api, 'dump', record_dump)
    results = [{'record': index, 'rows': list(range(600))}
               for index in range(10)]

    assert api.D13NNoReplaceDumpResults(
        str(target)).compute_metrics(results) == {}

    assert published_paths == [str(target)]
    assert dump_streams == [pending_stream]
    assert pending_stream.write_sizes
    restored = mmengine.load(
        io.BytesIO(bytes(pending_stream.payload)), file_format='pkl')
    assert restored == results


def test_dump_preserves_exact_order_for_240000_synthetic_rows(tmp_path):
    api = _api()
    target = tmp_path / 'large_predictions.pkl'
    results = [{
        'record': record,
        'rows': [(record, row) for row in range(600)],
    } for record in range(400)]

    api.D13NNoReplaceDumpResults(str(target)).compute_metrics(results)

    restored = mmengine.load(str(target))
    assert len(restored) == 400
    assert sum(len(item['rows']) for item in restored) == 240_000
    assert [item['record'] for item in restored] == list(range(400))
    assert [row for item in restored for row in item['rows']] == [
        (record, row) for record in range(400) for row in range(600)
    ]

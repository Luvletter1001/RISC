import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / 'tools' / 'risc_n0o' / 'openrsd_n0o_protocol.py')
V3_ROOT = (
    PROJECT_ROOT.parents[1] / 'docs' / 'provenance'
    / 'risc_openrsd_n0o_v3')


def load_protocol():
    module_name = 'openrsd_n0o_protocol'
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_v3(tmp_path):
    output = tmp_path / 'v3'
    shutil.copytree(V3_ROOT, output)
    return output


def test_loads_only_exact_v3_and_rejects_invalidation_or_drift(tmp_path):
    protocol = load_protocol()

    bundle = protocol.load_protocol_bundle(V3_ROOT)

    assert bundle.manifest_sha256 == protocol.V3_MANIFEST_SHA256
    assert len(bundle.scene_records) == 160
    assert len(bundle.support_rows) == 160
    assert set(bundle.scene_records) == set(bundle.support_rows)

    invalid = copy_v3(tmp_path / 'invalid')
    (invalid / 'INVALIDATED.json').write_text('{}\n')
    with pytest.raises(protocol.ProtocolError, match='INVALIDATED'):
        protocol.load_protocol_bundle(invalid)

    drift = copy_v3(tmp_path / 'drift')
    with (drift / 'support_ledger.jsonl').open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(protocol.ProtocolError, match='ledger hash'):
        protocol.load_protocol_bundle(drift)


def test_model_ledger_has_registered_view_order_and_information_firewall():
    protocol = load_protocol()
    bundle = protocol.load_protocol_bundle(V3_ROOT)

    rows = protocol.build_model_ledger(bundle)

    assert len(rows) == 160
    c4 = next(row for row in rows if row['fold_id'] == 'c4_a')
    c8 = next(row for row in rows if row['fold_id'] == 'c8_a')
    assert [view['view_id'] for view in c4['views']] == [
        'rot000_a', 'rot000_b', 'rot090', 'rot180', 'rot270']
    assert [view['view_id'] for view in c8['views']] == [
        'rot000_a', 'rot000_b', 'rot045', 'rot090', 'rot135',
        'rot180', 'rot225', 'rot270', 'rot315']
    assert c4['views'][0]['angle_deg'] == c4['views'][1]['angle_deg'] == 0
    for row in rows:
        sealed = bundle.support_rows[row['scene_id']]
        assert row['support_prompt_indices'] == [
            item['indices'] for item in sealed['selections']]
        assert len(row['support_prompt_indices']) == 18
        assert all(len(indices) == 7 for indices in row[
            'support_prompt_indices'])
    forbidden = ('annotation', 'gt', 'qbox', 'class', 'metric', 'prediction')
    for row in rows:
        encoded_keys = json.dumps(row, sort_keys=True).lower()
        assert not any(token in encoded_keys for token in forbidden)
    ledger_bytes = protocol.model_ledger_bytes(rows)
    assert ledger_bytes.endswith(b'\n')
    assert len(ledger_bytes.splitlines()) == 160


def test_support_cache_reconstructs_bitwise_scene_tensor(monkeypatch):
    protocol = load_protocol()
    bundle = protocol.load_protocol_bundle(V3_ROOT)
    monkeypatch.setattr(
        'torch.cuda.is_available',
        lambda: (_ for _ in ()).throw(AssertionError('CUDA queried')))

    cache = protocol.SupportCache.from_bundle(bundle)
    scene_id = next(iter(bundle.support_rows))
    support, labels = cache.for_scene(scene_id)

    assert tuple(support.shape) == (1, 126, 256)
    assert tuple(labels.shape) == (1, 126)
    assert support.device.type == labels.device.type == 'cpu'
    assert protocol.tensor_sha256(support[0].reshape(18, 7, 256)) == (
        bundle.support_rows[scene_id]['mapped_tensor_sha256'])
    assert labels[0].reshape(18, 7)[:, 0].tolist() == list(range(18))
    assert (labels[0].reshape(18, 7) == labels[0].reshape(18, 7)[:, :1]).all()

    report = protocol.audit_support_bundle(cache)
    assert report == {
        'scene_count': 160,
        'mapped_tensor_bundle_byte_count': 20643840,
        'mapped_tensor_bundle_sha256': (
            'e0daa61fd43f24746184139657b995820ba56c013d5c3caafd6fb9230f3c0fba'),
    }

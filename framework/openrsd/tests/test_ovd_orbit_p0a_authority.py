import json
import math
from pathlib import Path
from types import MappingProxyType

import pytest

import M_Tools.analysis.ovd_orbit_p0a_authority as authority_module
from M_Tools.analysis.ovd_orbit_p0a_authority import (
    P0AAuthorityError,
    build_authority_artifacts,
    canonical_json_bytes,
    load_canonical_json,
    load_inventory_rows,
    select_primary_objects,
    sha256_bytes,
    sha256_file,
    validate_inventory_receipt,
    validate_source_support_manifest,
)


ZERO_OBSERVED_CLASSES = ('container-crane', 'helicopter', 'helipad')
ANNOTATION_SHA256 = 'a' * 64


def _row(object_id, class_name, *, scene_id='P0001', size=100.0, overlap=0.05):
    return {
        'annotation_sha256': ANNOTATION_SHA256,
        'box': [10, 20, 30, 40, 0],
        'class_name': class_name,
        'object_id': object_id,
        'overlap': overlap,
        'scene_id': scene_id,
        'size': size,
    }


def _eligible_rows():
    rows = []
    for class_name in ('bridge', 'ship'):
        rows.extend(
            _row(f'{class_name}-{index}', class_name,
                 size=43.5 if index == 0 else 3265.0,
                 overlap=0.05)
            for index in range(5)
        )
    rows.extend(_row(f'rare-{index}', 'rare-class') for index in range(4))
    rows.extend((
        _row('too-small', 'bridge', size=43.49),
        _row('too-large', 'ship', size=3265.01),
        _row('too-overlapping', 'ship', overlap=0.0500001),
    ))
    return list(reversed(rows))


def test_selects_exact_fixed_policy_population_at_inclusive_bounds():
    selection = select_primary_objects(
        _eligible_rows(), zero_observed_classes=ZERO_OBSERVED_CLASSES)

    assert selection.primary_object_count == 10
    assert not hasattr(selection, 'primary_count')
    assert selection.supported_classes == ('bridge', 'ship')
    assert tuple(row['object_id'] for row in selection.primary_rows) == tuple(
        sorted(f'{class_name}-{index}'
               for class_name in ('bridge', 'ship') for index in range(5)))
    assert tuple(selection.policy) == (
        'area_min', 'area_max', 'max_overlap_iou', 'min_objects_per_class',
        'selection_basis',
    )
    assert dict(selection.policy) == {
        'area_min': 43.5,
        'area_max': 3265.0,
        'max_overlap_iou': 0.05,
        'min_objects_per_class': 5,
        'selection_basis': 'p0a_inventory_gt_quantiles_v1',
    }
    assert tuple(row['object_id'] for row in
                 selection.non_primary_diagnostics['unsupported_class']) == (
                     'rare-0', 'rare-1', 'rare-2', 'rare-3')
    assert tuple(row['object_id'] for row in
                 selection.non_primary_diagnostics['geometry_excluded']) == (
                     'too-large', 'too-overlapping', 'too-small')
    assert isinstance(selection.primary_rows[0], MappingProxyType)
    with pytest.raises(TypeError):
        selection.primary_rows[0]['size'] = 1
    with pytest.raises(TypeError):
        selection.policy['area_min'] = 0


def test_primary_rows_are_sorted_by_scene_then_object_id_after_input_reversal():
    rows = []
    for class_name in ('bridge', 'ship'):
        rows.extend((
            _row('z', class_name, scene_id='P0001'),
            _row('a', class_name, scene_id='P0002'),
            _row('y', class_name, scene_id='P0001'),
            _row('b', class_name, scene_id='P0002'),
            _row('x', class_name, scene_id='P0001'),
        ))
    for index, row in enumerate(rows):
        row['object_id'] = f'{row["object_id"]}-{row["class_name"]}'

    selection = select_primary_objects(
        list(reversed(rows)), zero_observed_classes=ZERO_OBSERVED_CLASSES)

    assert tuple((row['scene_id'], row['object_id'])
                 for row in selection.primary_rows) == (
                     ('P0001', 'x-bridge'), ('P0001', 'x-ship'),
                     ('P0001', 'y-bridge'), ('P0001', 'y-ship'),
                     ('P0001', 'z-bridge'), ('P0001', 'z-ship'),
                     ('P0002', 'a-bridge'), ('P0002', 'a-ship'),
                     ('P0002', 'b-bridge'), ('P0002', 'b-ship'),
                 )


@pytest.mark.parametrize('mutate', (
    lambda rows: rows.append(_row('bridge-0', 'bridge')),
    lambda rows: rows[0].update({'extra': 'not allowed'}),
    lambda rows: rows[0].update({'size': float('nan')}),
    lambda rows: rows[0].update({'overlap': float('inf')}),
    lambda rows: rows[0].update({'box': [0, 0, 1, 1, math.nan]}),
    lambda rows: rows[0].update({'box': [0, 0, 1, 1, True]}),
))
def test_rejects_invalid_or_ambiguous_inventory_rows(mutate):
    rows = _eligible_rows()
    mutate(rows)

    with pytest.raises(P0AAuthorityError):
        select_primary_objects(rows, zero_observed_classes=ZERO_OBSERVED_CLASSES)


@pytest.mark.parametrize('declaration', (
    pytest.param(None, id='missing'),
    pytest.param(['container-crane', 'helicopter', 'helipad'], id='list'),
    pytest.param(('container-crane', 'helipad', 'helicopter'), id='wrong-order'),
    pytest.param(('container-crane', 'helicopter'), id='incomplete'),
))
def test_requires_the_exact_zero_observed_class_declaration(declaration):
    rows = _eligible_rows()

    with pytest.raises(P0AAuthorityError):
        if declaration is None:
            select_primary_objects(rows)
        else:
            select_primary_objects(rows, zero_observed_classes=declaration)


def test_rejects_every_caller_policy_override():
    with pytest.raises(P0AAuthorityError):
        select_primary_objects(
            _eligible_rows(), zero_observed_classes=ZERO_OBSERVED_CLASSES,
            area_min=0)


def test_canonical_bytes_and_sha256_helpers_are_deterministic(tmp_path):
    payload = canonical_json_bytes({'z': '桥', 'a': [2, 1]})
    path = tmp_path / 'row.json'
    path.write_bytes(payload)

    assert payload == b'{"a":[2,1],"z":"\xe6\xa1\xa5"}\n'
    assert sha256_bytes(payload) == sha256_file(path)


FULL_CLASS_ORDER = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
)
CODE_PATHS = (
    'M_Tools/analysis/ovd_orbit_p0.py',
    'M_Tools/analysis/ovd_orbit_p0_export.py',
    'M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py',
)


@pytest.fixture(autouse=True)
def temporary_trusted_framework_root(tmp_path, monkeypatch):
    monkeypatch.setattr(authority_module, '_FRAMEWORK_ROOT', tmp_path)


def _write_canonical(path, value):
    path.write_bytes(canonical_json_bytes(value))


def _json_data(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _authority_inputs(tmp_path):
    inventory_rows = _eligible_rows()[:10]
    inventory_rows = [
        row for row in inventory_rows
        if row['class_name'] in {'bridge', 'ship'}
    ]
    # Make the deliberately small fixture valid for two five-object classes.
    inventory_rows = [
        _row(f'{class_name}-{index}', class_name)
        for class_name in ('bridge', 'ship') for index in range(5)
    ]
    inventory_path = tmp_path / 'object_inventory.jsonl'
    inventory_path.write_bytes(b''.join(
        canonical_json_bytes(row) for row in inventory_rows))
    diagnostics = {
        'schema': 'risc-n0-set-orbit-p0a-inventory-diagnostics-v1',
        'status': 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD',
        'retained_object_count': len(inventory_rows),
    }
    diagnostics_path = tmp_path / 'inventory_diagnostics.json'
    _write_canonical(diagnostics_path, diagnostics)
    candidate_path = tmp_path / 'candidate_scene_plan.json'
    _write_canonical(candidate_path, {'schema': 'test-candidate-v1', 'records': []})
    result_path = tmp_path / 'result.md'
    result_path.write_bytes(b'opaque inventory receipt\n')
    receipt_path = tmp_path / 'receipt.json'
    _write_canonical(receipt_path, {
        'schema': 'risc-n0-set-orbit-p0a-inventory-receipt-v1',
        'status': 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD',
        'artifact_sha256': {
            'candidate_scene_plan.json': sha256_file(candidate_path),
            'inventory_diagnostics.json': sha256_file(diagnostics_path),
            'object_inventory.jsonl': sha256_file(inventory_path),
            'result.md': sha256_file(result_path),
        },
    })
    support_path = tmp_path / 'opaque_text7_support.bin'
    support_path.write_bytes(b'opaque support bytes; never deserialize\n')
    manifest_path = tmp_path / 'source_input_manifest.json'
    _write_canonical(manifest_path, {
        'schema': 'risc-openrsd-n0o-input-manifest-v1',
        'support': {
            'class_count': 18,
            'class_order': list(FULL_CLASS_ORDER),
            'shot': 7,
            'source': {
                'path': str(support_path),
                'sha256': sha256_file(support_path),
            },
            'support_type': 'text',
        },
    })
    code_paths = []
    for relative_path in CODE_PATHS:
        code_path = tmp_path / relative_path
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_bytes(f'opaque source {relative_path}\n'.encode('utf-8'))
        code_paths.append(code_path)
    return {
        'receipt': receipt_path,
        'diagnostics': diagnostics_path,
        'inventory': inventory_path,
        'manifest': manifest_path,
        'support': support_path,
        'code': tuple(code_paths),
    }


def _build_authority(inputs):
    return build_authority_artifacts(
        inventory_receipt_path=inputs['receipt'],
        inventory_diagnostics_path=inputs['diagnostics'],
        object_inventory_path=inputs['inventory'],
        source_input_manifest_path=inputs['manifest'],
        support_asset_path=inputs['support'],
        oracle_code_files=inputs['code'],
    )


def test_snapshot_bound_authority_artifacts_are_deterministic(tmp_path):
    inputs = _authority_inputs(tmp_path)

    artifacts = _build_authority(inputs)

    assert tuple(artifacts) == (
        'p0a_diagnostic_authority.json', 'primary_object_ids.jsonl',
        'authority_diagnostics.json', 'receipt.json', 'result.md')
    authority = json.loads(artifacts['p0a_diagnostic_authority.json'])
    assert authority['condition_id'] == 'historical_text7_support_v1'
    assert authority['prompt_stability_status'] == 'NOT_TESTED_SINGLE_CONDITION'
    assert authority['condition'] == {'support_type': 'text', 'shot': 7}
    assert authority['support_asset_path'] == str(inputs['support'])
    assert authority['support_asset_sha256'] == sha256_file(inputs['support'])
    assert authority['support_type'] == 'text'
    assert authority['shot'] == 7
    assert authority['oracle_adapter_type'] == 'dense-carrier-fallback'
    assert authority['source_schema'] == 'level-row-v1'
    assert authority['score_field'] == 'raw_native_foreground_logits'
    assert 'base' not in authority
    assert 'novel' not in authority
    assert [item['sha256'] for item in authority['oracle_code_identities']] == [
        sha256_file(path) for path in inputs['code']]
    receipt = json.loads(artifacts['receipt.json'])
    assert receipt['artifact_sha256'] == {
        name: sha256_bytes(payload)
        for name, payload in artifacts.items() if name != 'receipt.json'
    }
    assert artifacts == _build_authority(inputs)


def test_support_validation_uses_the_one_captured_snapshot_not_a_second_read(
        tmp_path, monkeypatch):
    inputs = _authority_inputs(tmp_path)
    original_support_sha256 = sha256_file(inputs['support'])
    original_decode_rows = authority_module._decode_inventory_rows

    def mutate_support_after_all_snapshots(raw, digest):
        inputs['support'].write_bytes(b'mutated after snapshot\n')
        return original_decode_rows(raw, digest)

    monkeypatch.setattr(
        authority_module, '_decode_inventory_rows', mutate_support_after_all_snapshots)

    artifacts = _build_authority(inputs)
    generated = json.loads(artifacts['p0a_diagnostic_authority.json'])

    assert sha256_file(inputs['support']) != original_support_sha256
    assert generated['support_asset_sha256'] == original_support_sha256


@pytest.mark.parametrize('mutate', (
    pytest.param(
        lambda inputs: _write_canonical(inputs['receipt'], {
            **_json_data(inputs['receipt']),
            'artifact_sha256': {
                **_json_data(inputs['receipt'])['artifact_sha256'],
                'object_inventory.jsonl': '0' * 64,
            },
        }), id='inventory-receipt-artifact-sha-mismatch'),
    pytest.param(
        lambda inputs: inputs['diagnostics'].write_bytes(b'{"bad":true}\n'),
        id='inventory-diagnostics-sha-mismatch'),
    pytest.param(
        lambda inputs: inputs['support'].write_bytes(b'mutated opaque support\n'),
        id='support-byte-mismatch'),
    pytest.param(
        lambda inputs: _write_canonical(inputs['manifest'], {
            **_json_data(inputs['manifest']),
            'support': {
                **_json_data(inputs['manifest'])['support'],
                'class_order': list(reversed(FULL_CLASS_ORDER)),
            },
        }), id='source-manifest-class-order-mismatch'),
    pytest.param(
        lambda inputs: inputs.update(
            code=(inputs['code'][0], inputs['code'][0], inputs['code'][2])),
        id='code-path-or-identity-mismatch'),
    pytest.param(
        lambda inputs: _replace_diagnostics_count(inputs, 11),
        id='wrong-retained-object-count-assertion'),
))
def test_rejects_each_snapshot_or_assertion_mismatch(tmp_path, mutate):
    inputs = _authority_inputs(tmp_path)
    mutate(inputs)

    with pytest.raises(P0AAuthorityError):
        _build_authority(inputs)


def _replace_diagnostics_count(inputs, count):
    diagnostics = _json_data(inputs['diagnostics'])
    diagnostics['retained_object_count'] = count
    _write_canonical(inputs['diagnostics'], diagnostics)


def test_snapshot_loaders_and_support_manifest_return_validated_values(tmp_path):
    inputs = _authority_inputs(tmp_path)
    receipt = load_canonical_json(inputs['receipt'], label='receipt')
    diagnostics = load_canonical_json(inputs['diagnostics'], label='diagnostics')
    rows = load_inventory_rows(inputs['inventory'])

    validate_inventory_receipt(
        receipt, inventory_diagnostics=diagnostics,
        object_inventory_rows=rows,
        candidate_scene_plan_path=tmp_path / 'candidate_scene_plan.json',
        inventory_result_path=tmp_path / 'result.md')
    support = validate_source_support_manifest(
        load_canonical_json(inputs['manifest'], label='manifest'),
        support_asset_path=inputs['support'])

    assert len(rows) == 10
    assert support['class_order'] == FULL_CLASS_ORDER


def test_canonical_snapshot_data_and_hash_are_immutably_bound(tmp_path):
    inputs = _authority_inputs(tmp_path)
    receipt = load_canonical_json(inputs['receipt'], label='receipt')
    original_digest = receipt.source_sha256
    inputs['receipt'].write_bytes(b'{"replacement":true}\n')

    with pytest.raises(TypeError):
        receipt.data['status'] = 'mutated'
    with pytest.raises(TypeError):
        receipt.data['artifact_sha256']['result.md'] = '0' * 64
    with pytest.raises(AttributeError):
        receipt.source_sha256 = '0' * 64

    assert receipt.source_sha256 == original_digest
    assert receipt.data['status'] == 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD'


@pytest.mark.parametrize('mutate', (
    pytest.param(
        lambda inputs, tmp_path: _replace_with_spoofed_suffix(inputs, tmp_path),
        id='spoofed-suffix'),
    pytest.param(
        lambda inputs, tmp_path: _replace_with_outside_root(inputs, tmp_path),
        id='outside-trusted-root'),
    pytest.param(
        lambda inputs, tmp_path: inputs.update(
            code=(inputs['code'][0], inputs['code'][0], inputs['code'][2])),
        id='duplicate'),
    pytest.param(
        lambda inputs, tmp_path: inputs.update(
            code=(inputs['code'][1], inputs['code'][0], inputs['code'][2])),
        id='wrong-order'),
    pytest.param(
        lambda inputs, tmp_path: _replace_with_symlink(
            inputs['code'][0], tmp_path / 'outside-code.py'), id='symlink'),
))
def test_oracle_code_inputs_must_be_exact_trusted_root_files(tmp_path, mutate):
    inputs = _authority_inputs(tmp_path)
    if 'spoof' in str(inputs['code'][0]):
        pytest.skip('sanity guard')
    mutate(inputs, tmp_path)

    with pytest.raises(P0AAuthorityError):
        _build_authority(inputs)


def test_oracle_snapshot_rejects_a_symlink_swap_immediately_before_open(
        tmp_path, monkeypatch):
    inputs = _authority_inputs(tmp_path)
    target = tmp_path / 'swap-target.py'
    target.write_bytes(b'swapped external source\n')
    original_open = authority_module.os.open
    swapped = False

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (not swapped and path == 'ovd_orbit_p0.py'
                and dir_fd is not None):
            inputs['code'][0].unlink()
            inputs['code'][0].symlink_to(target)
            swapped = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(authority_module.os, 'open', swap_before_open)

    with pytest.raises(P0AAuthorityError):
        _build_authority(inputs)
    assert swapped


def test_oracle_snapshot_rejects_an_intermediate_directory_symlink_swap(
        tmp_path, monkeypatch):
    inputs = _authority_inputs(tmp_path)
    original_open = authority_module.os.open
    swapped = False

    def swap_intermediate(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == 'M_Tools' and dir_fd is not None and not swapped:
            original_tree = tmp_path / 'M_Tools.original'
            (tmp_path / 'M_Tools').rename(original_tree)
            external_tree = tmp_path / 'external' / 'M_Tools' / 'analysis'
            external_tree.mkdir(parents=True)
            (external_tree / 'ovd_orbit_p0.py').write_bytes(
                b'external swapped source\n')
            (tmp_path / 'M_Tools').symlink_to(
                external_tree.parent, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(authority_module.os, 'open', swap_intermediate)

    with pytest.raises(P0AAuthorityError):
        _build_authority(inputs)
    assert swapped


def _replace_with_symlink(path, target):
    target.write_bytes(b'outside trusted root\n')
    path.unlink()
    path.symlink_to(target)


def _replace_with_spoofed_suffix(inputs, tmp_path):
    spoofed = tmp_path / 'spoof' / CODE_PATHS[0]
    spoofed.parent.mkdir(parents=True, exist_ok=True)
    spoofed.write_bytes(b'spoofed source\n')
    inputs['code'] = (spoofed, inputs['code'][1], inputs['code'][2])


def _replace_with_outside_root(inputs, tmp_path):
    outside = tmp_path.parent / 'outside-root' / CODE_PATHS[0]
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b'outside source\n')
    inputs['code'] = (outside, inputs['code'][1], inputs['code'][2])


def test_rejects_self_consistent_receipt_with_out_of_vocabulary_row(tmp_path):
    inputs = _authority_inputs(tmp_path)
    rows = [json.loads(line) for line in inputs['inventory'].read_text().splitlines()]
    rows[0]['class_name'] = 'not-in-diagnostic-vocabulary'
    inputs['inventory'].write_bytes(b''.join(canonical_json_bytes(row) for row in rows))
    diagnostics = dict(load_canonical_json(
        inputs['diagnostics'], label='diagnostics').data)
    diagnostics['retained_object_count'] = len(rows)
    _write_canonical(inputs['diagnostics'], diagnostics)
    receipt = dict(load_canonical_json(inputs['receipt'], label='receipt').data)
    receipt['artifact_sha256'] = {
        'candidate_scene_plan.json': sha256_file(
            tmp_path / 'candidate_scene_plan.json'),
        'inventory_diagnostics.json': sha256_file(inputs['diagnostics']),
        'object_inventory.jsonl': sha256_file(inputs['inventory']),
        'result.md': sha256_file(tmp_path / 'result.md'),
    }
    _write_canonical(inputs['receipt'], receipt)

    with pytest.raises(P0AAuthorityError):
        _build_authority(inputs)

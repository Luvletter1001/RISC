import math
from copy import deepcopy
import json
from pathlib import Path

import pytest

import M_Tools.analysis.ovd_orbit_p0a_inventory as inventory
from M_Tools.analysis.ovd_orbit_p0a_inventory import (
    P0AInventoryError,
    build_inventory_artifacts,
    build_inventory_failure_artifacts,
    canonical_json_bytes,
    convex_iou,
    load_canonical_json,
    normalize_rbox,
    parse_dota_line,
    polygon_area,
    sha256_bytes,
    sha256_file,
    validate_source_manifest,
    validate_source_plan,
)


VOCABULARY = ('bridge',)
ANNOTATION_SHA256 = 'a' * 64


def test_canonical_json_bytes_is_sorted_compact_utf8_and_rejects_nan():
    assert canonical_json_bytes({'z': '桥', 'a': [2, 1]}) == (
        b'{"a":[2,1],"z":"\xe6\xa1\xa5"}\n')
    with pytest.raises(ValueError):
        canonical_json_bytes({'value': float('nan')})


def test_sha256_helpers_return_the_same_digest_for_bytes_and_a_file(tmp_path):
    payload = b'P0-A inventory contract\n'
    path = tmp_path / 'annotation.txt'
    path.write_bytes(payload)

    assert sha256_bytes(payload) == sha256_file(path)
    assert sha256_bytes(payload) == (
        'd9b40645b6fb356a51bcaab53f4733877f7775029d9748a4697490d991840009')


def test_normalize_rbox_orders_sides_and_wraps_theta():
    assert normalize_rbox([2, 1, 2, 4, 0]) == pytest.approx(
        [2, 1, 4, 2, -math.pi / 2])


def test_normalize_rbox_rejects_a_non_sequence_box():
    with pytest.raises(P0AInventoryError):
        normalize_rbox(None)


def test_normalize_rbox_rejects_a_numeric_conversion_overflow():
    with pytest.raises(P0AInventoryError):
        normalize_rbox([0, 0, 10**400, 1, 0])


def test_polygon_area_and_convex_iou_return_zero_when_disjoint():
    first = ((0, 0), (4, 0), (4, 2), (0, 2))
    second = ((5, 0), (7, 0), (7, 2), (5, 2))

    assert polygon_area(first) == pytest.approx(8)
    assert convex_iou(first, second) == 0


def test_parse_dota_line_builds_the_canonical_first_object():
    object_record = parse_dota_line(
        '0 0 4 0 4 2 0 2 bridge 1',
        line_number=1,
        scene_id='P0001',
        annotation_sha256=ANNOTATION_SHA256,
        vocabulary=VOCABULARY,
    )

    assert object_record['object_id'] == 'P0001:1'
    assert object_record['box'] == pytest.approx([2, 1, 4, 2, 0])
    assert object_record['size'] == pytest.approx(8)
    assert object_record['difficulty'] == 1


def test_parse_dota_line_exposes_polygon_for_convex_iou():
    first = parse_dota_line(
        '0 0 4 0 4 2 0 2 bridge 1',
        line_number=1,
        scene_id='P0001',
        annotation_sha256=ANNOTATION_SHA256,
        vocabulary=VOCABULARY,
    )
    second = parse_dota_line(
        '2 0 6 0 6 2 2 2 bridge 0',
        line_number=2,
        scene_id='P0001',
        annotation_sha256=ANNOTATION_SHA256,
        vocabulary=VOCABULARY,
    )

    assert convex_iou(first['polygon'], second['polygon']) == pytest.approx(1 / 3)


def test_parse_dota_line_accepts_finite_tiny_geometry_and_self_iou():
    object_record = parse_dota_line(
        '0 0 0.0000001 0 0.0000001 0.0000001 0 0.0000001 bridge 0',
        line_number=1,
        scene_id='P0001',
        annotation_sha256=ANNOTATION_SHA256,
        vocabulary=VOCABULARY,
    )

    assert convex_iou(object_record['polygon'], object_record['polygon']) == pytest.approx(1)


def test_parse_dota_line_rejects_overflowed_derived_geometry_with_line_context():
    with pytest.raises(P0AInventoryError, match=r'line 1'):
        parse_dota_line(
            '0 0 1e308 0 1e308 1e308 0 1e308 bridge 0',
            line_number=1,
            scene_id='P0001',
            annotation_sha256=ANNOTATION_SHA256,
            vocabulary=VOCABULARY,
        )


@pytest.mark.parametrize('line', (
    '0 0 4 0 4 2 0 2 bridge',
    'nan 0 4 0 4 2 0 2 bridge 1',
    '0 0 4 0 4 2 0 2 ship 1',
    '0 0 4 0 4 2 0 2 bridge 3',
    '0 0 0 0 0 2 0 2 bridge 1',
))
def test_parse_dota_line_rejects_invalid_input_with_line_number(line):
    with pytest.raises(P0AInventoryError, match=r'line 1'):
        parse_dota_line(
            line,
            line_number=1,
            scene_id='P0001',
            annotation_sha256=ANNOTATION_SHA256,
            vocabulary=VOCABULARY,
        )


FULL_VOCABULARY = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    'container-crane', 'airport', 'helipad',
)


def canonical_source_inputs(tmp_path):
    """Create the sealed 160-record source plan using temporary annotations."""
    records = []
    for fold_id, offset in (
            ('c4_a', 1000), ('c4_b', 2000), ('c8_a', 3000), ('c8_b', 4000)):
        for scene_rank in range(1, 41):
            scene_id = f'P{offset + scene_rank:04d}'
            annotation_path = tmp_path / f'{scene_id}.txt'
            lines = [
                '0 0 4 0 4 2 0 2 bridge 0',
                '2 0 6 0 6 2 2 2 bridge 1',
            ]
            if fold_id == 'c4_a' and scene_rank == 1:
                lines.append('8 0 12 0 12 2 8 2 bridge 2')
            annotation_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            records.append({
                'fold_id': fold_id,
                'scene_id': scene_id,
                'scene_rank': scene_rank,
                # Deliberately missing: no inventory function may open or hash images.
                'image_path': str(tmp_path / 'missing-images' / f'{scene_id}.png'),
                'image_sha256': '1' * 64,
                'annotation_path': str(annotation_path),
                'annotation_sha256': sha256_file(annotation_path),
            })
    return {
        'schema': 'risc-n0-set-orbit-scene-plan-v1',
        'unique_scene_count': 160,
        'records': records,
    }, {'support': {'class_order': list(FULL_VOCABULARY)}}


def build_artifacts_for(source_plan, source_manifest):
    return build_inventory_artifacts(
        source_plan=source_plan,
        source_plan_sha256=sha256_bytes(canonical_json_bytes(source_plan)),
        source_input_manifest=source_manifest,
    )


def test_inventory_artifacts_build_the_c4_diagnostic_package(tmp_path):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)

    artifacts = build_artifacts_for(source_plan, source_manifest)
    candidates = json.loads(artifacts['candidate_scene_plan.json'])
    rows = [json.loads(line) for line in artifacts['object_inventory.jsonl'].splitlines()]
    diagnostics = json.loads(artifacts['inventory_diagnostics.json'])
    receipt = json.loads(artifacts['receipt.json'])

    assert set(artifacts) == {
        'candidate_scene_plan.json', 'object_inventory.jsonl',
        'inventory_diagnostics.json', 'receipt.json', 'result.md',
    }
    assert len(candidates['records']) == 80
    assert {row['split'] for row in candidates['records']} == {'diagnostic'}
    assert all(row['scene_id'] != 'P0148' for row in candidates['records'])
    assert all(set(row) == {
        'annotation_sha256', 'box', 'class_name', 'object_id', 'overlap',
        'scene_id', 'size',
    } for row in rows)
    assert all('difficulty' not in row and 'polygon' not in row for row in rows)
    first_scene_rows = [row for row in rows if row['scene_id'] == 'P1001']
    assert [row['overlap'] for row in first_scene_rows] == pytest.approx([1 / 3, 1 / 3])
    assert diagnostics['selected_scene_count'] == 80
    assert diagnostics['retained_object_count'] == 160
    assert diagnostics['difficulty_2_object_count'] == 1
    assert diagnostics['annotation_verification_count'] == 80
    assert diagnostics['class_counts'] == {'bridge': 160}
    assert diagnostics['size_quantiles'] == {
        '0.0': 8.0, '0.25': 8.0, '0.5': 8.0, '0.75': 8.0, '1.0': 8.0,
    }
    assert diagnostics['overlap_quantiles'] == pytest.approx({
        '0.0': 1 / 3, '0.25': 1 / 3, '0.5': 1 / 3, '0.75': 1 / 3,
        '1.0': 1 / 3,
    })
    assert receipt['status'] == 'P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD'
    assert receipt['artifact_sha256'] == {
        name: sha256_bytes(artifacts[name])
        for name in sorted(artifacts) if name != 'receipt.json'
    }
    assert '模型指标' not in artifacts['result.md'].decode('utf-8')


def test_inventory_artifacts_are_byte_deterministic_when_source_records_reverse(tmp_path):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)
    reversed_source_plan = deepcopy(source_plan)
    reversed_source_plan['records'].reverse()

    assert build_artifacts_for(source_plan, source_manifest) == build_artifacts_for(
        reversed_source_plan, source_manifest)


def test_inventory_parses_the_exact_annotation_snapshot_bytes(tmp_path, monkeypatch):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)
    first_annotation = source_plan['records'][0]['annotation_path']

    def snapshot_then_mutate(path):
        payload = Path(path).read_bytes()
        digest = sha256_bytes(payload)
        if str(path) == first_annotation:
            Path(path).write_text(
                '0 0 4 0 4 2 0 2 unexpected 0\n', encoding='utf-8')
        return payload, digest

    monkeypatch.setattr(inventory, 'read_annotation_snapshot', snapshot_then_mutate)

    artifacts = build_artifacts_for(source_plan, source_manifest)

    diagnostics = json.loads(artifacts['inventory_diagnostics.json'])
    assert diagnostics['annotation_verification_count'] == 80


@pytest.mark.parametrize('mutation', (
    'p0148', 'bad_plan_sha', 'duplicate_c4', 'only_79_c4', 'annotation_mismatch',
    'unknown_class', 'short_manifest',
))
def test_inventory_contract_violations_fail_closed(tmp_path, mutation):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)
    if mutation == 'p0148':
        source_plan['records'][0]['scene_id'] = 'P0148'
    elif mutation == 'duplicate_c4':
        source_plan['records'][1]['scene_id'] = source_plan['records'][0]['scene_id']
    elif mutation == 'only_79_c4':
        source_plan['records'].pop(0)
    elif mutation == 'annotation_mismatch':
        source_plan['records'][0]['annotation_sha256'] = '0' * 64
    elif mutation == 'unknown_class':
        path = source_plan['records'][0]['annotation_path']
        payload = '0 0 4 0 4 2 0 2 unexpected 0\n'
        Path(path).write_text(payload, encoding='utf-8')
        source_plan['records'][0]['annotation_sha256'] = sha256_bytes(payload.encode())
    elif mutation == 'short_manifest':
        source_manifest['support']['class_order'].pop()

    plan_sha = sha256_bytes(canonical_json_bytes(source_plan))
    if mutation == 'bad_plan_sha':
        plan_sha = '0' * 64
    with pytest.raises(P0AInventoryError):
        build_inventory_artifacts(
            source_plan=source_plan,
            source_plan_sha256=plan_sha,
            source_input_manifest=source_manifest,
        )


def test_source_plan_and_manifest_validators_return_only_the_selected_contract(tmp_path):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)

    selected = validate_source_plan(source_plan)

    assert len(selected) == 80
    assert [row['fold_id'] for row in selected[:2]] == ['c4_a', 'c4_a']
    assert selected[0]['scene_rank'] == 1
    assert validate_source_manifest(source_manifest) == FULL_VOCABULARY


def test_source_plan_rejects_a_truncated_80_c4_plan_declared_as_160(tmp_path):
    source_plan, _ = canonical_source_inputs(tmp_path)
    source_plan['records'] = [
        record for record in source_plan['records']
        if record['fold_id'] in {'c4_a', 'c4_b'}
    ]

    with pytest.raises(P0AInventoryError, match=r'length.*160'):
        validate_source_plan(source_plan)


def test_source_plan_rejects_a_160_record_plan_with_only_79_c4_scenes(tmp_path):
    source_plan, source_manifest = canonical_source_inputs(tmp_path)
    source_plan['records'][0]['fold_id'] = 'c8_a'

    with pytest.raises(P0AInventoryError, match=r'exactly 80 C4'):
        build_artifacts_for(source_plan, source_manifest)


def test_source_plan_requires_an_integer_unique_scene_count(tmp_path):
    source_plan, _ = canonical_source_inputs(tmp_path)
    source_plan['unique_scene_count'] = 160.0

    with pytest.raises(P0AInventoryError, match=r'unique_scene_count'):
        validate_source_plan(source_plan)


def test_quantiles_use_deterministic_linear_interpolation_for_size_and_overlap():
    expected = {'0.0': 0.0, '0.25': 3.0, '0.5': 6.0, '0.75': 9.0, '1.0': 12.0}

    assert inventory._quantiles([0.0, 4.0, 8.0, 12.0], 'size') == expected
    assert inventory._quantiles([12.0, 0.0, 8.0, 4.0], 'overlap') == expected


def test_load_canonical_json_retains_labeled_source_snapshot_digest(tmp_path):
    value = {'support': {'class_order': list(FULL_VOCABULARY)}}
    path = tmp_path / 'source.json'
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)

    snapshot = load_canonical_json(path, label='source input manifest')

    assert snapshot == value
    assert snapshot.source_sha256 == sha256_bytes(payload)


def test_failure_artifacts_are_a_minimal_chinese_fail_stop_package():
    artifacts = build_inventory_failure_artifacts(P0AInventoryError('source plan mismatch'))

    assert set(artifacts) == {
        'receipt.json', 'inventory_diagnostics.json', 'result.md',
    }
    assert json.loads(artifacts['receipt.json'])['status'] == 'P0_INPUT_FAIL_STOP'
    assert '模型指标' not in artifacts['result.md'].decode('utf-8')

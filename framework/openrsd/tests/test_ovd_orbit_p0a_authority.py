import math
from types import MappingProxyType

import pytest

from M_Tools.analysis.ovd_orbit_p0a_authority import (
    P0AAuthorityError,
    canonical_json_bytes,
    select_primary_objects,
    sha256_bytes,
    sha256_file,
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

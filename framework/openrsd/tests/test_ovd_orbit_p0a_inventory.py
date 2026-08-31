import math

import pytest

from M_Tools.analysis.ovd_orbit_p0a_inventory import (
    P0AInventoryError,
    canonical_json_bytes,
    convex_iou,
    normalize_rbox,
    parse_dota_line,
    polygon_area,
    sha256_bytes,
    sha256_file,
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

import math

from M_Tools.analysis.build_dotav2_annfiles_from_uploaded_tiles import (
    DotaObject,
    parse_tile_stem,
    project_objects_to_tile,
)


def test_parse_tile_stem_reads_source_size_and_offsets():
    tile = parse_tile_stem("P0000__1024__1048___2096")

    assert tile.source_id == "P0000"
    assert tile.tile_size == 1024
    assert tile.x == 1048
    assert tile.y == 2096


def test_project_objects_scales_raw_polygon_and_translates_to_tile():
    obj = DotaObject(
        poly=[(1100.0, 2100.0), (1200.0, 2100.0), (1200.0, 2200.0), (1100.0, 2200.0)],
        label="plane",
        difficulty="0",
    )

    lines = project_objects_to_tile([obj], parse_tile_stem("P0000__1024__1048___2096"))

    assert len(lines) == 1
    parts = lines[0].split()
    coords = [float(v) for v in parts[:8]]
    assert parts[8:] == ["plane", "0"]
    expected = [52.0, 4.0, 152.0, 4.0, 152.0, 104.0, 52.0, 104.0]
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(coords, expected))


def test_project_objects_marks_partial_objects_as_difficult():
    obj = DotaObject(
        poly=[(900.0, 2100.0), (1200.0, 2100.0), (1200.0, 2200.0), (900.0, 2200.0)],
        label="ship",
        difficulty="0",
    )

    lines = project_objects_to_tile(
        [obj],
        parse_tile_stem("P0000__1024__1048___2096"),
        iof_thr=0.4,
    )

    assert len(lines) == 1
    assert lines[0].split()[8:] == ["ship", "2"]

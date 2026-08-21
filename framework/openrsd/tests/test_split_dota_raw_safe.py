from M_Tools.analysis.split_dota_raw_safe_1024 import (
    clip_polygon_to_rect,
    get_sliding_windows,
    polygon_area,
    project_objects_to_window,
)


def test_polygon_clip_to_half_rect():
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    clipped = clip_polygon_to_rect(poly, (0, 0, 5, 10))

    assert polygon_area(poly) == 100.0
    assert polygon_area(clipped) == 50.0


def test_get_sliding_windows_one_exact_tile():
    assert get_sliding_windows(1024, 1024, 1024, 500, 0.6) == [(0, 0, 1024, 1024)]


def test_project_objects_to_window_translates_and_marks_truncated():
    objects = [
        {
            "poly": [(500.0, 500.0), (1500.0, 500.0), (1500.0, 700.0), (500.0, 700.0)],
            "label": "ship",
            "diff": "0",
        }
    ]

    lines = project_objects_to_window(objects, (0, 0, 1024, 1024), iof_thr=0.1)

    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[8] == "ship"
    assert parts[9] == "2"
    assert max(float(value) for value in parts[0::2][:4]) > 1024.0

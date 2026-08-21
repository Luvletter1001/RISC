import csv

from M_Tools.analysis.build_ep2_clean_path_probe_from_logits import (
    build_path_probe_rows,
    build_path_probe_from_logits,
)


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_path_probe_rows_extracts_class_logits_and_z_values():
    raw = [{
        "case_id": "c1",
        "control_type": "clean",
        "context_id": "ctx",
        "object_id": "obj",
        "pair": "small-vehicle->plane",
        "path_step": "0",
        "log_area": "4.6",
        "logit::small-vehicle": "1.2",
        "logit::plane": "2.4",
        "z::small-vehicle": "0.5",
        "z::plane": "-4.0",
    }]

    rows = build_path_probe_rows(raw)

    assert rows == [{
        "case_id": "c1",
        "pair": "small-vehicle->plane",
        "gt_class": "small-vehicle",
        "hardneg_class": "plane",
        "control_type": "clean",
        "context_id": "ctx",
        "object_id": "obj",
        "path_step": "0",
        "log_area": "4.6",
        "logit_gt": "1.2",
        "logit_hardneg": "2.4",
        "z_gt": "0.5",
        "z_hardneg": "-4.0",
        "abs_z_gt": "0.5",
        "abs_z_hardneg": "4.0",
        "hardneg_margin": "1.2",
        "support_advantage": "-3.5",
    }]


def test_build_path_probe_from_logits_writes_auditable_csv(tmp_path):
    raw_csv = tmp_path / "raw_logits.csv"
    out_csv = tmp_path / "path_probe.csv"
    write_csv(raw_csv, [
        {
            "case_id": "c1",
            "control_type": "clean",
            "context_id": "ctx",
            "object_id": "obj",
            "gt_class": "ship",
            "hardneg_class": "harbor",
            "path_step": "0",
            "logit_ship": "0.2",
            "logit_harbor": "0.8",
            "abs_z_ship": "1.0",
            "abs_z_harbor": "3.0",
        }
    ])

    result = build_path_probe_from_logits(raw_csv, out_csv)

    assert result["rows_written"] == 1
    rows = read_csv(out_csv)
    assert rows[0]["gt_class"] == "ship"
    assert rows[0]["hardneg_class"] == "harbor"
    assert rows[0]["hardneg_margin"] == "0.6"
    assert rows[0]["support_advantage"] == "-2.0"

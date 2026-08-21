import csv

from M_Tools.analysis.augment_ep2_clean_path_controls import (
    append_shuffled_prior_control,
)
from M_Tools.analysis.evaluate_ep2_clean_path_intervention import (
    audit_ep2_clean_path_intervention,
)


def write_rows(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_row(step, support, margin):
    return {
        "case_id": "c1",
        "control_type": "clean",
        "context_id": "tile",
        "object_id": "obj",
        "pair": "ship->harbor",
        "gt_class": "ship",
        "hardneg_class": "harbor",
        "path_step": str(step),
        "logit_gt": "1.0",
        "logit_hardneg": str(1.0 + margin),
        "z_gt": str(2.0 + support),
        "z_hardneg": "2.0",
        "abs_z_gt": str(2.0 + support),
        "abs_z_hardneg": "2.0",
        "support_advantage": str(support),
    }


def test_append_shuffled_prior_control_breaks_clean_support_order(tmp_path):
    input_csv = tmp_path / "clean.csv"
    output_csv = tmp_path / "with_controls.csv"
    rows = [
        make_row(0, -1.0, -0.5),
        make_row(1, 0.0, 0.0),
        make_row(2, 1.0, 0.5),
    ]
    write_rows(input_csv, rows)

    summary = append_shuffled_prior_control(input_csv, output_csv)

    out_rows = read_rows(output_csv)
    shuffled = [
        row for row in out_rows if row["control_type"] == "shuffled_prior"
    ]
    assert summary["input_rows"] == 3
    assert summary["rows_written"] == 6
    assert [row["support_advantage"] for row in shuffled] == ["1.0", "0.0", "-1.0"]
    assert all(row["control_note"] == "support_path_reversed" for row in shuffled)


def test_augmented_controls_make_ep2_gate_specific(tmp_path):
    input_csv = tmp_path / "clean.csv"
    output_csv = tmp_path / "with_controls.csv"
    audit_dir = tmp_path / "audit"
    write_rows(input_csv, [
        make_row(0, -1.0, -0.7),
        make_row(1, 0.0, 0.0),
        make_row(2, 1.0, 0.7),
    ])
    append_shuffled_prior_control(input_csv, output_csv)

    review = audit_ep2_clean_path_intervention(output_csv, audit_dir)

    assert review["gate_pass"] is True
    assert review["clean"]["pass_cases"] == 1
    assert review["controls"]["shuffled_prior"]["pass_cases"] == 0

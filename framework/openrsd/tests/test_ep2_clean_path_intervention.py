import csv
import json

import pytest

from M_Tools.analysis.evaluate_ep2_clean_path_intervention import (
    audit_ep2_clean_path_intervention,
    summarize_case,
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


def make_row(case_id, step, support_advantage, margin, control_type="clean",
             context_id="tile_001", object_id="crop_001"):
    return {
        "case_id": case_id,
        "pair": "small-vehicle->plane",
        "gt_class": "small-vehicle",
        "hardneg_class": "plane",
        "control_type": control_type,
        "context_id": context_id,
        "object_id": object_id,
        "path_step": str(step),
        "abs_z_gt": str(2.0 + support_advantage),
        "abs_z_hardneg": "2.0",
        "logit_gt": "1.0",
        "logit_hardneg": str(1.0 + margin),
    }


def test_summarize_case_passes_when_support_advantage_tracks_margin():
    rows = [
        make_row("c1", 0, -1.0, -0.7),
        make_row("c1", 1, 0.0, 0.0),
        make_row("c1", 2, 1.0, 0.7),
    ]

    summary = summarize_case(rows, min_points=3, min_rho=0.8,
                             min_margin_delta=0.5,
                             min_support_delta=1.5)

    assert summary["status"] == "pass"
    assert summary["fixed_context"] is True
    assert summary["fixed_object"] is True
    assert summary["rho_support_margin"] == pytest.approx(1.0)
    assert summary["margin_delta"] == pytest.approx(1.4)


def test_audit_rejects_control_even_when_clean_case_passes(tmp_path):
    input_csv = tmp_path / "path_probe.csv"
    out_dir = tmp_path / "out"
    rows = [
        make_row("c1", 0, -1.0, -0.7, "clean"),
        make_row("c1", 1, 0.0, 0.0, "clean"),
        make_row("c1", 2, 1.0, 0.7, "clean"),
        make_row("c1", 0, -1.0, 0.2, "shuffled_prior"),
        make_row("c1", 1, 0.0, 0.1, "shuffled_prior"),
        make_row("c1", 2, 1.0, 0.0, "shuffled_prior"),
    ]
    write_rows(input_csv, rows)

    review = audit_ep2_clean_path_intervention(input_csv, out_dir)

    assert review["gate_pass"] is True
    assert review["clean"]["pass_cases"] == 1
    assert review["controls"]["shuffled_prior"]["pass_cases"] == 0
    assert review["specificity_gap"] == pytest.approx(1.0)
    saved = json.loads((out_dir / "ep2_clean_review.json").read_text())
    assert saved["gate_pass"] is True
    assert (out_dir / "ep2_clean_case_summary.csv").exists()
    assert (out_dir / "ep2_clean_report.md").exists()


def test_varying_context_marks_case_confounded():
    rows = [
        make_row("c2", 0, -1.0, -0.7, context_id="tile_a"),
        make_row("c2", 1, 0.0, 0.0, context_id="tile_b"),
        make_row("c2", 2, 1.0, 0.7, context_id="tile_c"),
    ]

    summary = summarize_case(rows)

    assert summary["status"] == "confounded_context_or_object"
    assert summary["fixed_context"] is False
    assert summary["pass_gate"] is False

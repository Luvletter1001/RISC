import csv
import json

from M_Tools.analysis.build_wrong_class_run_summary import build_summary_row
from M_Tools.analysis.build_wrong_class_run_summary import write_summary_csv


def test_build_summary_row_reads_latest_eval_metrics_and_wrong_stats(tmp_path):
    output_dir = tmp_path / "run"
    old_dir = output_dir / "test_work" / "20260617_010000"
    new_dir = output_dir / "test_work" / "20260617_020000"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "20260617_010000.json").write_text(
        json.dumps({"dota/mAP": 0.1, "dota/AP50": 0.2})
    )
    new_eval = new_dir / "20260617_020000.json"
    new_eval.write_text(json.dumps({"dota/mAP": 0.6535041332, "dota/AP50": 0.654}))

    wrong_summary = output_dir / "wrong_class_iou_gt0p7_summary.json"
    wrong_summary.write_text(
        json.dumps(
            {
                "stats": {
                    "images_scanned": 9772,
                    "detections_checked": 214793,
                    "gt_objects_checked": 87403,
                    "localized_correct_class": 66089,
                },
                "selected_count": 8241,
                "out_jsonl": str(output_dir / "wrong.jsonl"),
                "out_csv": str(output_dir / "wrong.csv"),
            }
        )
    )

    row = build_summary_row(
        model="model_a",
        config="config.py",
        checkpoint="epoch.pth",
        output_dir=output_dir,
        wrong_summary_json=wrong_summary,
    )

    assert row["mAP"] == "0.6535"
    assert row["AP50"] == "0.654"
    assert row["images_scanned"] == "9772"
    assert row["wrong_class_iou_gt0p7"] == "8241"
    assert row["summary_json"] == str(wrong_summary)


def test_write_summary_csv_uses_existing_baseline_field_order(tmp_path):
    row = {
        "model": "model_a",
        "config": "config.py",
        "checkpoint": "epoch.pth",
        "output_dir": "run",
        "mAP": "0.1",
        "AP50": "0.2",
        "images_scanned": "1",
        "detections_checked": "2",
        "gt_objects_checked": "3",
        "localized_correct_class": "4",
        "wrong_class_iou_gt0p7": "5",
        "jsonl": "wrong.jsonl",
        "csv": "wrong.csv",
        "summary_json": "summary.json",
    }
    out_csv = tmp_path / "summary.csv"
    write_summary_csv(out_csv, [row])

    with out_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == [
        "model",
        "config",
        "checkpoint",
        "output_dir",
        "mAP",
        "AP50",
        "images_scanned",
        "detections_checked",
        "gt_objects_checked",
        "localized_correct_class",
        "wrong_class_iou_gt0p7",
        "jsonl",
        "csv",
        "summary_json",
    ]
    assert rows == [row]

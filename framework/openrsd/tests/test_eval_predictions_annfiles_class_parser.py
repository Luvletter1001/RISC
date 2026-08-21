from pathlib import Path

from M_Tools.analysis.eval_predictions_annfiles_metric_json import (
    DOTA1_CLASSES,
    load_class_names,
    parse_classes_from_config_text,
)


def test_parse_classes_ignores_rotation_agnostic_classes(tmp_path: Path):
    config = tmp_path / "h2rbox_eval.py"
    config.write_text(
        """
model = dict(
    bbox_head=dict(
        num_classes=15,
        rotation_agnostic_classes=[1, 9, 11],
    )
)
""",
        encoding="utf-8",
    )

    assert parse_classes_from_config_text(config) == DOTA1_CLASSES


def test_load_class_names_uses_dotav2_classes_for_dotav2_dataset(tmp_path: Path):
    config = tmp_path / "dotav2_eval.py"
    config.write_text(
        """
test_dataloader = dict(dataset=dict(type='DOTAv2Dataset'))
""",
        encoding="utf-8",
    )

    assert load_class_names(str(config)) == (
        "airport",
        "baseball-diamond",
        "basketball-court",
        "bridge",
        "container-crane",
        "ground-track-field",
        "harbor",
        "helicopter",
        "helipad",
        "large-vehicle",
        "plane",
        "roundabout",
        "ship",
        "small-vehicle",
        "soccer-ball-field",
        "storage-tank",
        "swimming-pool",
        "tennis-court",
    )

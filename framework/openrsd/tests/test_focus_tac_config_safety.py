from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch.nn as nn

from M_AD.engine.runner.meta_remove_runer import apply_trainable_parameter_filter


SCRIPT_PATH = Path(
    "experiments/rotation_semantic_attractor/scripts/"
    "100_focus_tac_generate_configs.py")


def _load_generator():
    spec = importlib.util.spec_from_file_location("focus_tac_generate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["focus_tac_generate"] = module
    spec.loader.exec_module(module)
    return module


def _generate(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    mod = _load_generator()
    output_dir = tmp_path / "configs"
    rc = mod.main([
        "--repo-root", str(Path.cwd()),
        "--focus-checkpoint",
        "work_dirs/focus_ovd_a10_sv_only_dota2_recovery_full_gpu69_20260609/"
        "epoch_24.pth",
        "--base-config",
        "M_configs/experiments/focus_ovd/"
        "focus_ovd_a10_sv_only_dota2_recovery_full.py",
        "--output-dir", str(output_dir),
        "--no-prefer-ema-state",
    ])
    assert rc == 0
    index_path = output_dir / "focus_tac_variant_config_index.csv"
    with index_path.open("r", encoding="utf-8", newline="") as f:
        return output_dir, list(csv.DictReader(f))


def test_generated_configs_do_not_enable_forbidden_support_changes(tmp_path):
    output_dir, rows = _generate(tmp_path)
    assert {row["variant_id"] for row in rows} == {
        "TAC_V00_FOCUS_ep24_eval",
        "TAC_V01_FOCUS_TAC_zero",
        "TAC_V02_FOCUS_TAC_alpha_only_all_classes",
        "TAC_V03_FOCUS_TAC_alpha_only_monitored",
        "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
    }

    for config_path in output_dir.glob("*.py"):
        text = config_path.read_text(encoding="utf-8")
        assert "support_type='visual'" not in text
        assert "support_type = 'visual'" not in text
        assert "use_declip_support=True" not in text
        assert "use_declip_support = True" not in text
        assert "with_aux_bbox_head=False" not in text
        assert "with_aux_bbox_head = False" not in text
        assert "dual_fusion=dict" not in text
        assert "eqtext=dict" not in text
        assert (
            "trainable_parameters = ['bbox_head.focus_text_anchor_calibration']"
            in text)


def test_training_variants_allow_only_tac_parameters(tmp_path):
    _output_dir, rows = _generate(tmp_path)
    train_rows = {
        row["variant_id"]: row for row in rows if row["train_required"] == "true"
    }
    assert set(train_rows) == {
        "TAC_V02_FOCUS_TAC_alpha_only_all_classes",
        "TAC_V03_FOCUS_TAC_alpha_only_monitored",
        "TAC_V04_FOCUS_TAC_alpha_beta_monitored",
    }

    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.bbox_head = nn.Module()
    model.bbox_head.focus_support_adapter = nn.Linear(2, 2)
    model.bbox_head.focus_text_anchor_calibration = nn.Module()
    model.bbox_head.focus_text_anchor_calibration.raw_alpha = nn.Parameter(
        model.backbone.weight.detach().new_zeros(4))
    model.bbox_head.focus_text_anchor_calibration.raw_beta = nn.Parameter(
        model.backbone.weight.detach().new_zeros(4))

    summary = apply_trainable_parameter_filter(
        model,
        trainable_substrings=("bbox_head.focus_text_anchor_calibration",),
    )
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    assert trainable == {
        "bbox_head.focus_text_anchor_calibration.raw_alpha",
        "bbox_head.focus_text_anchor_calibration.raw_beta",
    }
    assert set(summary["trainable"]) == trainable
    assert "bbox_head.focus_support_adapter.weight" in summary["frozen"]


def test_eval_variants_are_marked_eval_only(tmp_path):
    output_dir, rows = _generate(tmp_path)
    flags = {row["variant_id"]: row["train_required"] for row in rows}

    assert flags["TAC_V00_FOCUS_ep24_eval"] == "false"
    assert flags["TAC_V01_FOCUS_TAC_zero"] == "false"
    assert flags["TAC_V03_FOCUS_TAC_alpha_only_monitored"] == "true"
    assert flags["TAC_V04_FOCUS_TAC_alpha_beta_monitored"] == "true"

    for variant_id in (
            "TAC_V00_FOCUS_ep24_eval",
            "TAC_V01_FOCUS_TAC_zero"):
        text = (output_dir / f"{variant_id}.py").read_text(encoding="utf-8")
        assert "custom_hooks = []" in text
        assert "FocusOVDTrainableAuditHook" not in text

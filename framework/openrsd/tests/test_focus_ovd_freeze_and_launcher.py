import json
from pathlib import Path

import torch.nn as nn

from M_AD.engine.runner.meta_remove_runer import apply_trainable_parameter_filter
from M_Tools.experiments.focus_ovd_wait_and_launch_gpu69 import run_preflight


class TinyFocusModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.neck = nn.Linear(2, 2)
        self.bbox_head = nn.Module()
        self.bbox_head.focus_support_adapter = nn.Linear(2, 2)
        self.bbox_head.rtm_cls = nn.Linear(2, 2)
        self.bbox_head.rtm_reg = nn.Linear(2, 2)


def test_trainable_allowlist_keeps_only_focus_adapter_trainable():
    model = TinyFocusModel()

    summary = apply_trainable_parameter_filter(
        model,
        trainable_substrings=("bbox_head.focus_support_adapter",),
    )

    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    assert trainable == {
        "bbox_head.focus_support_adapter.weight",
        "bbox_head.focus_support_adapter.bias",
    }
    assert summary["trainable_count"] == 2
    assert summary["frozen_count"] == 8


def test_watcher_preflight_missing_checkpoint_is_fatal(tmp_path):
    full_config = tmp_path / "full.py"
    smoke_config = tmp_path / "smoke.py"
    missing_checkpoint = tmp_path / "missing.pth"
    result_dir = tmp_path / "logs"
    full_config.write_text("model = dict()\n", encoding="utf-8")
    smoke_config.write_text("model = dict()\n", encoding="utf-8")

    status = run_preflight(
        full_config=full_config,
        smoke_config=smoke_config,
        checkpoint=missing_checkpoint,
        result_dir=result_dir,
        work_dir_full=tmp_path / "work-full",
        work_dir_smoke=tmp_path / "work-smoke",
    )

    assert status.ok is False
    assert status.code == "FATAL_MISSING_CHECKPOINT"
    status_json = result_dir / "preflight_status.json"
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert payload["code"] == "FATAL_MISSING_CHECKPOINT"
    assert str(missing_checkpoint) in payload["message"]

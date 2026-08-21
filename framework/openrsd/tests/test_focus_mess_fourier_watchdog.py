from pathlib import Path

from M_Tools.experiments.focus_mess_fourier_watchdog import (
    analyze_summary,
    can_launch_autotune,
    parse_focus_log_text,
    render_autotune_config,
)


SAMPLE_LOG = """
06/11 00:04:32 - mmengine - INFO - Epoch(val) [6][551/551]    dota/mAP: 0.5491  dota/AP50: 0.5490  dota/IoU_50_Detail: {'small-vehicle': {'ap': 0.1069, 'recall': 0.3654, 'num_dets': 3028050, 'num_gts': 150145}, 'plane': {'ap': 0.8803, 'recall': 0.9346, 'num_dets': 184277, 'num_gts': 8718}}  data_time: 0.0041  time: 0.3952
06/11 00:20:51 - mmengine - INFO - Epoch(train)  [9][ 50/400]  base_lr: 2.5000e-04 lr: 2.5000e-04  eta: 1:04:47  time: 0.6590  data_time: 0.0086  memory: 16178  loss: 2021.7783  loss_aln: 0.3207  loss_cls: 2020.0865  loss_bbox: 0.4456
06/11 00:21:23 - mmengine - INFO - Epoch(train)  [9][100/400]  base_lr: 2.5000e-04 lr: 2.5000e-04  eta: 1:04:18  time: 0.6343  data_time: 0.0061  memory: 16177  loss: 2546.7252  loss_aln: 0.3172  loss_cls: 2544.9973  loss_bbox: 0.4510
06/11 00:22:23 - mmengine - INFO - Epoch(train)  [9][200/400]  base_lr: 2.5000e-04 lr: 2.5000e-04  eta: 1:03:14  time: 0.5926  data_time: 0.0055  memory: 15326  loss: 2439.7024  loss_aln: 0.3287  loss_cls: 2437.9474  loss_bbox: 0.4359
06/11 00:28:23 - mmengine - INFO - Epoch(train) [10][400/400]  base_lr: 2.5000e-04 lr: 2.5000e-04  eta: 0:56:53  time: 0.6468  data_time: 0.0066  memory: 14479  loss: 2124.8932  loss_aln: 0.3114  loss_cls: 2123.4104  loss_bbox: 0.3433
06/11 00:36:11 - mmengine - INFO - Epoch(val) [10][551/551]    dota/mAP: 0.5300  dota/AP50: 0.5300  dota/IoU_50_Detail: {'small-vehicle': {'ap': 0.0500, 'recall': 0.3600, 'num_dets': 3500000, 'num_gts': 150145}, 'plane': {'ap': 0.8700, 'recall': 0.9300, 'num_dets': 184277, 'num_gts': 8718}}  data_time: 0.0041  time: 0.3952
"""


def test_parse_focus_log_text_extracts_train_and_val_metrics():
    summary = parse_focus_log_text(SAMPLE_LOG, max_epochs=24)

    assert summary.latest_train_epoch == 10
    assert summary.latest_train_iter == 400
    assert summary.best_val.epoch == 6
    assert summary.best_val.map == 0.5491
    assert summary.final_val.epoch == 10
    assert summary.final_val.small_vehicle_ap == 0.05
    assert summary.final_val.small_vehicle_num_dets == 3500000
    assert not summary.has_error
    assert not summary.is_complete


def test_analyze_summary_triggers_conservative_rerun_for_val_drop_and_overdetect():
    summary = parse_focus_log_text(SAMPLE_LOG, max_epochs=10)

    decision = analyze_summary(summary)

    assert decision.should_rerun
    assert decision.strategy == "reduce_text_overdetect"
    assert decision.text_weight_init == 0.01
    assert decision.max_text_weight == 0.01
    assert any("mAP drop" in reason for reason in decision.reasons)
    assert any("small-vehicle AP drop" in reason for reason in decision.reasons)
    assert any("over-detection" in reason for reason in decision.reasons)


def test_render_autotune_config_keeps_conservative_caps(tmp_path):
    summary = parse_focus_log_text(SAMPLE_LOG, max_epochs=10)
    decision = analyze_summary(summary)
    config_text = render_autotune_config(
        base_config=Path("M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py"),
        work_dir=Path("work_dirs/focus_autotune_r1"),
        report_dir=Path("resultmd/exp_focus_ovd_20260608/mess_fourier_dual_text_6gpu/autotune_r1"),
        round_index=1,
        decision=decision,
    )

    assert "_base_ = './focus_ovd_a10_mess_fourier_dual_text_6gpu.py'" in config_text
    assert "work_dir = 'work_dirs/focus_autotune_r1'" in config_text
    assert "focus_text_anchor_calibration=dict(enable=False)" in config_text
    assert "focus_fourier_head_gate=dict(enable=False)" in config_text
    assert "focus_text_logit_mixer=dict(enable=False)" in config_text
    assert "text_weight_init=0.01" in config_text
    assert "max_text_weight=0.01" in config_text
    assert "alpha_t_max=0.015" in config_text
    assert "alpha_m_max=0.015" in config_text


def test_final_train_epoch_without_final_val_is_not_complete():
    text = """
06/11 02:01:00 - mmengine - INFO - Epoch(train) [24][400/400]  loss: 2200.0  loss_cls: 2199.0
06/11 02:01:30 - mmengine - INFO - Epoch(val) [24][ 50/551]    eta: 0:03:10  time: 0.3900
"""
    summary = parse_focus_log_text(text, max_epochs=24)

    assert not summary.is_complete
    assert analyze_summary(summary).strategy == "wait_for_validation"


def test_can_launch_autotune_blocks_on_telemetry_failure_or_active_process():
    summary = parse_focus_log_text(SAMPLE_LOG, max_epochs=10)
    decision = analyze_summary(summary)

    telemetry_gate = can_launch_autotune(
        decision=decision,
        summary=summary,
        telemetry_ok=False,
        session_alive=False,
        active_processes=[],
    )
    active_gate = can_launch_autotune(
        decision=decision,
        summary=summary,
        telemetry_ok=True,
        session_alive=False,
        active_processes=[
            "python -m torch.distributed.run --standalone tools/train.py "
            "M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py"
        ],
    )

    assert not telemetry_gate.allowed
    assert "telemetry" in telemetry_gate.reason
    assert not active_gate.allowed
    assert "active training process" in active_gate.reason


def test_plain_nccl_config_text_is_not_a_fatal_error():
    text = """
06/11 00:00:00 - mmengine - INFO - dist_cfg=dict(backend='nccl')
06/11 00:00:01 - mmengine - INFO - Epoch(val) [24][551/551]    dota/mAP: 0.5600  dota/AP50: 0.5600  dota/IoU_50_Detail: {'small-vehicle': {'ap': 0.1200, 'recall': 0.3900, 'num_dets': 2800000, 'num_gts': 150145}}  data_time: 0.0041
"""

    summary = parse_focus_log_text(text, max_epochs=24)

    assert not summary.has_error
    assert summary.is_complete

from pathlib import Path

from mmengine.config import Config

from M_AD.engine.runner.meta_remove_runer import apply_trainable_parameter_filter


ZERO_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_mess_fourier_dual_text_6gpu_zero.py")
TRAIN_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_mess_fourier_dual_text_6gpu.py")
LAUNCHER = Path("M_Tools/experiments/run_focus_mess_fourier_dual_text_6gpu.sh")
GPU67_BEST_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_mess_fourier_dual_text_gpu67_best.py")
GPU67_BEST_BS1_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_mess_fourier_dual_text_gpu67_best_bs1.py")
GPU67_LOWTEXT_BEST_BS1_CONFIG = Path(
    "M_configs/experiments/focus_ovd/"
    "focus_ovd_a10_mess_fourier_dual_text_gpu67_lowtext_best_bs1.py")
GPU67_TRAIN_EVAL_LAUNCHER = Path(
    "M_Tools/experiments/run_focus_mess_fourier_dual_text_gpu67_best_train_eval.sh")


def test_dual_text_zero_config_is_six_gpu_and_disables_failed_logit_modules():
    cfg = Config.fromfile(ZERO_CONFIG)
    bbox_head = cfg.model["bbox_head"]
    focus = bbox_head["focus_ovd"]

    assert cfg.num_gpus == 6
    assert cfg.batch_size == 1
    assert cfg.train_dataloader["batch_size"] == 1
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 6
    assert cfg.train_dataloader["sampler"]["batch_size"] == 1
    assert cfg.trainable_parameters == [
        "bbox_head.focus_support_adapter",
        "bbox_head.focus_text_adapter",
        "bbox_head.focus_dual_support_fusion",
    ]
    assert bbox_head["focus_text_anchor_calibration"]["enable"] is False
    assert bbox_head["focus_fourier_head_gate"]["enable"] is False
    assert bbox_head["focus_text_logit_mixer"]["enable"] is False
    assert focus["eqtext"]["type"] == "mess_fourier_dual"
    assert focus["eqtext"]["mess_branch"]["group_order"] == 8
    assert focus["dual_fusion"]["max_text_weight"] == 0.0
    assert (
        cfg.model["support_feat_dict"]["Data1_DOTA2"] ==
        "./data/DOTA2_1024_500/ss_train/"
        "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
    assert (
        cfg.train_dataloader["dataset"]["datasets"][0]["data_root"] ==
        "data/DOTA2_1024_500/ss_train")


def test_dual_text_train_config_keeps_text_weight_cap_at_two_percent():
    cfg = Config.fromfile(TRAIN_CONFIG)
    bbox_head = cfg.model["bbox_head"]
    focus = bbox_head["focus_ovd"]
    audit_hook = next(
        hook for hook in cfg.custom_hooks
        if hook["type"] == "FocusOVDTrainableAuditHook")

    assert cfg.num_gpus == 6
    assert cfg.batch_size == 1
    assert cfg.train_dataloader["batch_size"] == 1
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 6
    assert cfg.train_dataloader["sampler"]["batch_size"] == 1
    assert cfg.trainable_parameters == [
        "bbox_head.focus_support_adapter",
        "bbox_head.focus_text_adapter",
        "bbox_head.focus_dual_support_fusion",
    ]
    assert bbox_head["focus_text_anchor_calibration"]["enable"] is False
    assert bbox_head["focus_fourier_head_gate"]["enable"] is False
    assert bbox_head["focus_text_logit_mixer"]["enable"] is False
    assert focus["eqtext"]["type"] == "mess_fourier_dual"
    assert focus["eqtext"]["mess_branch"]["group_order"] == 8
    assert focus["dual_fusion"]["max_text_weight"] == 0.02
    assert (
        cfg.model["support_feat_dict"]["Data1_DOTA2"] ==
        "./data/DOTA2_1024_500/ss_train/"
        "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl")
    assert (
        cfg.train_dataloader["dataset"]["datasets"][0]["data_root"] ==
        "data/DOTA2_1024_500/ss_train")
    assert "focus_trainable_audit.json" in audit_hook["log_path"]
    assert "focus_trainable_audit_zero.json" not in audit_hook["log_path"]


def test_dual_text_trainable_allowlist_keeps_only_focus_modules_trainable():
    import torch.nn as nn

    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.bbox_head = nn.Module()
    model.bbox_head.focus_support_adapter = nn.Linear(2, 2)
    model.bbox_head.focus_text_adapter = nn.Linear(2, 2)
    model.bbox_head.focus_dual_support_fusion = nn.Linear(2, 2)
    model.bbox_head.focus_text_logit_mixer = nn.Linear(2, 2)
    model.bbox_head.focus_fourier_head_gate = nn.Linear(2, 2)
    model.bbox_head.focus_text_anchor_calibration = nn.Linear(2, 2)

    summary = apply_trainable_parameter_filter(
        model,
        trainable_substrings=[
            "bbox_head.focus_support_adapter",
            "bbox_head.focus_text_adapter",
            "bbox_head.focus_dual_support_fusion",
        ])
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    assert trainable == {
        "bbox_head.focus_support_adapter.weight",
        "bbox_head.focus_support_adapter.bias",
        "bbox_head.focus_text_adapter.weight",
        "bbox_head.focus_text_adapter.bias",
        "bbox_head.focus_dual_support_fusion.weight",
        "bbox_head.focus_dual_support_fusion.bias",
    }
    assert all("focus_text_logit_mixer" not in name for name in summary["trainable"])
    assert all("focus_fourier_head_gate" not in name for name in summary["trainable"])
    assert all("focus_text_anchor_calibration" not in name for name in summary["trainable"])


def test_dual_text_launcher_disables_unreliable_a40_nccl_transports():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "export NCCL_P2P_DISABLE=1" in text
    assert "export NCCL_IB_DISABLE=1" in text
    assert "export PYTHONPATH=\"${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}\"" in text
    assert "--standalone" in text
    assert "--nproc_per_node=\"${NPROC}\"" in text
    assert "NPROC=${NPROC:-6}" in text
    assert "0,1,2,3,4,5" in text
    assert (
        "M_configs/experiments/focus_ovd/"
        "focus_ovd_a10_mess_fourier_dual_text_6gpu.py"
    ) in text
    assert "focus_ovd_a10_mess_fourier_dual_text_6gpu.py" in text


def test_dual_text_launcher_is_executable():
    assert LAUNCHER.stat().st_mode & 0o111


def test_dual_text_gpu67_best_config_uses_two_cards_and_saves_best_map():
    cfg = Config.fromfile(GPU67_BEST_CONFIG)
    checkpoint_hook = cfg.default_hooks["checkpoint"]
    audit_hook = next(
        hook for hook in cfg.custom_hooks
        if hook["type"] == "FocusOVDTrainableAuditHook")

    assert cfg.num_gpus == 2
    assert cfg.batch_size == 2
    assert cfg.train_dataloader["batch_size"] == 2
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 2
    assert cfg.train_dataloader["sampler"]["batch_size"] == 2
    assert cfg.model["bbox_head"]["focus_ovd"]["dual_fusion"]["max_text_weight"] == 0.01
    assert cfg.model["bbox_head"]["focus_ovd"]["eqtext"]["alpha_t_max"] == 0.015
    assert cfg.model["bbox_head"]["focus_ovd"]["eqtext"]["mess_branch"]["alpha_m_max"] == 0.015
    assert checkpoint_hook["type"] == "CheckpointHook"
    assert checkpoint_hook["save_best"] == "dota/mAP"
    assert checkpoint_hook["rule"] == "greater"
    assert checkpoint_hook["max_keep_ckpts"] == 3
    assert "gpu67_best" in audit_hook["log_path"]


def test_gpu67_train_eval_launcher_uses_physical_gpu67_and_best_checkpoint():
    text = GPU67_TRAIN_EVAL_LAUNCHER.read_text(encoding="utf-8")

    assert "GPUS=${GPUS:-6,7}" in text
    assert "NPROC=${NPROC:-2}" in text
    assert "export NCCL_P2P_DISABLE=1" in text
    assert "export NCCL_IB_DISABLE=1" in text
    assert "best_*.pth" in text
    assert "last_checkpoint" not in text
    assert "tools/openrsd_test.py" in text
    assert "--nproc_per_node=\"${NPROC}\"" in text
    assert "focus_ovd_a10_mess_fourier_dual_text_gpu67_best.py" in text


def test_dual_text_gpu67_best_bs1_config_is_low_memory_full_24epoch():
    cfg = Config.fromfile(GPU67_BEST_BS1_CONFIG)
    checkpoint_hook = cfg.default_hooks["checkpoint"]
    audit_hook = next(
        hook for hook in cfg.custom_hooks
        if hook["type"] == "FocusOVDTrainableAuditHook")

    assert cfg.num_gpus == 2
    assert cfg.batch_size == 1
    assert cfg.max_epochs == 24
    assert cfg.train_cfg["max_epochs"] == 24
    assert cfg.train_dataloader["batch_size"] == 1
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 2
    assert cfg.train_dataloader["sampler"]["batch_size"] == 1
    assert checkpoint_hook["save_best"] == "dota/mAP"
    assert checkpoint_hook["rule"] == "greater"
    assert checkpoint_hook["max_keep_ckpts"] == 3
    assert "gpu67_best_bs1" in cfg.work_dir
    assert "gpu67_best_bs1" in audit_hook["log_path"]


def test_dual_text_gpu67_lowtext_config_reduces_text_intervention():
    cfg = Config.fromfile(GPU67_LOWTEXT_BEST_BS1_CONFIG)
    focus = cfg.model["bbox_head"]["focus_ovd"]
    eqtext = focus["eqtext"]
    checkpoint_hook = cfg.default_hooks["checkpoint"]
    audit_hook = next(
        hook for hook in cfg.custom_hooks
        if hook["type"] == "FocusOVDTrainableAuditHook")

    assert cfg.num_gpus == 2
    assert cfg.batch_size == 1
    assert cfg.max_epochs == 24
    assert cfg.train_dataloader["batch_size"] == 1
    assert cfg.train_dataloader["sampler"]["num_gpus"] == 2
    assert cfg.train_dataloader["sampler"]["batch_size"] == 1
    assert eqtext["alpha_t_max"] == 0.008
    assert eqtext["max_delta_norm_ratio"] == 0.01
    assert eqtext["mess_branch"]["alpha_m_max"] == 0.008
    assert eqtext["mess_branch"]["low_rank"] == 8
    assert eqtext["fourier_branch"]["alpha_t_max"] == 0.008
    assert eqtext["fourier_branch"]["low_rank"] == 8
    assert eqtext["text_branch_fusion"]["mess_weight_init"] == 0.25
    assert eqtext["text_branch_fusion"]["fourier_weight_init"] == 0.75
    assert eqtext["text_branch_fusion"]["max_branch_weight"] == 0.25
    assert focus["dual_fusion"]["visual_weight_init"] == 0.995
    assert focus["dual_fusion"]["text_weight_init"] == 0.005
    assert focus["dual_fusion"]["max_text_weight"] == 0.005
    assert checkpoint_hook["save_best"] == "dota/mAP"
    assert checkpoint_hook["rule"] == "greater"
    assert "gpu67_lowtext_best_bs1" in cfg.work_dir
    assert "gpu67_lowtext_best_bs1" in audit_hook["log_path"]

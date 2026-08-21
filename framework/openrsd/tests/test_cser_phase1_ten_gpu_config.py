from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "M_configs/Diagnostics/"
    "dota1_cser_phase1_frozen400_gpu0123456789_20260803.py"
)
LAUNCHER_PATH = (
    REPO_ROOT
    / "scripts/research_queues/"
    "run_dota1_cser_phase1_frozen400_gpu0123456789_20260803.sh"
)


def test_phase1_config_has_frozen_adapter_only_contract():
    cfg = Config.fromfile(CONFIG_PATH)

    assert cfg.load_from == (
        "/data1/zcy/OpenRSD_results/results/"
        "MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth"
    )
    assert cfg.resume is False
    assert cfg.trainable_parameters == ["bbox_head.counter_support_ratio"]

    adapter = cfg.model.bbox_head.counter_support_ratio
    assert adapter.enable is True
    assert adapter.rank == 8
    assert adapter.init_strength == 0.0
    assert cfg.model.support_type == "visual"
    assert cfg.model.num_val_prompts == 8


def test_phase1_config_matches_ten_gpu_sampler_and_data_contract():
    cfg = Config.fromfile(CONFIG_PATH)
    train = cfg.train_dataloader
    sampler = train.sampler

    assert train.batch_size == 2
    assert "batch_sampler" not in train
    assert sampler.batch_size == train.batch_size
    assert sampler.num_gpus == 10
    assert sampler.max_iter_per_epoch == 400
    assert sampler.source_prob == [1.0]
    assert cfg.train_cfg.max_epochs == 1

    datasets = train.dataset.datasets
    assert len(datasets) == 1
    assert datasets[0].dataset_flag == "Data1_DOTA2"
    assert datasets[0].ann_file.endswith("train_formatted_pkls")
    assert cfg.val_dataloader.dataset.ann_file == "annfiles"
    assert cfg.val_dataloader.dataset.data_prefix.img_path == "images"


def test_phase1_config_freezes_norm_stats_without_ema():
    cfg = Config.fromfile(CONFIG_PATH)
    hook_types = [hook.type for hook in cfg.custom_hooks]

    assert hook_types == ["FreezeNormStatsHook"]
    assert cfg.default_hooks.logger.interval <= 10
    assert cfg.default_hooks.checkpoint.interval == 1


def test_phase1_launcher_uses_exact_ten_gpu_nccl_contract():
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9" in launcher
    assert "NCCL_P2P_DISABLE=1" in launcher
    assert "NCCL_IB_DISABLE=1" in launcher
    assert "--nproc_per_node=10" in launcher
    assert "--master_port=29683" in launcher


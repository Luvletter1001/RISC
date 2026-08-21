import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from M_Tools.rotation_sv_repair.common import SuiteContext
from M_Tools.rotation_sv_repair.paper_method_stacks import build_paper_repair_stack
from M_Tools.rotation_sv_repair.rotation_adapter import (
    LowRankRotationAdapter,
    apply_rotation_adapter_to_feats,
    resolve_adapter_ckpt,
    save_adapter,
)
def _mini_ctx(work_dir: Path) -> SuiteContext:
    return SuiteContext(
        repo_root=work_dir.parent,
        work_dir=work_dir,
        result_dir=work_dir / 'result',
        gpu_ids='0',
        mode='smoke',
        exp='adapter',
        teacher='ensemble',
        student='adapter',
        pseudo_label_mode='dynamic_queue',
        vocab_mode='dynamic_vocabulary',
        teacher_vlm='available_auto',
    )


def test_apply_rotation_adapter_changes_spatial_feat():
    adapter = LowRankRotationAdapter(8, rank=2, alpha=0.5)
    with torch.no_grad():
        adapter.up.weight.fill_(0.02)
    x = torch.randn(2, 8, 4, 4)
    y = apply_rotation_adapter_to_feats(x, adapter)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_build_paper_stack_loads_ckpt():
    with TemporaryDirectory() as td:
        work = Path(td)
        ctx = _mini_ctx(work)
        ckpt_dir = ctx.adapter_ckpt_dir
        adapter = LowRankRotationAdapter(256, rank=4, alpha=0.3)
        save_adapter(adapter, ckpt_dir / 'adapter_best_sv_repair.pth', meta=dict(rank=4, alpha=0.3))
        assert resolve_adapter_ckpt(ctx) is not None
        stack = build_paper_repair_stack('lowrank_r16_a0.3', ctx)
        assert getattr(stack, 'rotation_adapter', None) is not None
        assert stack.calibration is None


def test_full_method_stack_has_adapter_and_calibration():
    with TemporaryDirectory() as td:
        work = Path(td)
        ctx = _mini_ctx(work)
        adapter = LowRankRotationAdapter(256, rank=4, alpha=0.3)
        save_adapter(adapter, ctx.adapter_ckpt_dir / 'adapter_latest.pth', meta=dict(rank=4, alpha=0.3))
        stack = build_paper_repair_stack('full_method', ctx)
        assert stack.rotation_adapter is not None
        assert stack.calibration is not None

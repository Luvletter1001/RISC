import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from M_Tools.rotation_sv_repair.rotation_adapter import LowRankRotationAdapter
from M_Tools.rotation_sv_repair.rotation_adapter import load_adapter, save_adapter


def test_adapter_alpha_zero_identity():
    a = LowRankRotationAdapter(32, rank=4, alpha=0.0)
    x = torch.randn(3, 32)
    assert torch.allclose(a(x), x)


def test_adapter_save_load_output_consistent():
    torch.manual_seed(7)
    x = torch.randn(3, 8)
    adapter = LowRankRotationAdapter(8, rank=2, alpha=0.5)
    with torch.no_grad():
        adapter.up.weight.fill_(0.01)
    y0 = adapter(x)
    with TemporaryDirectory() as td:
        path = Path(td) / 'adapter.pth'
        save_adapter(adapter, path, meta={'dim': 8, 'rank': 2, 'alpha': 0.5})
        loaded = load_adapter(path, lambda: LowRankRotationAdapter(8, rank=2, alpha=0.5))
    y1 = loaded(x)
    assert torch.allclose(y0, y1, atol=1e-6)

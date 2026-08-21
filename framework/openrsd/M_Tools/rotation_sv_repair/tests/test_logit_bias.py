import torch
from tools.sv_attractor_repair_gpu89.repair_modules import ClassWiseCalibration

def test_logit_bias_only_sv():
    cal = ClassWiseCalibration(15)
    cal.set_sv_bias_only(10, -0.5)
    x = torch.zeros(4, 15)
    y = cal(x)
    assert torch.allclose(y[:, :10], x[:, :10])
    assert torch.allclose(y[:, 10], x[:, 10] - 0.5)
    assert torch.allclose(y[:, 11:], x[:, 11:])

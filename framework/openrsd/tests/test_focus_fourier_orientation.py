import math

import torch

from M_AD.models.utils.focus_fourier_orientation import FourierOrientationLearner


def _oriented_sinusoid(angle_rad, size=33, cycles=4.0):
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    direction = math.cos(angle_rad) * xx + math.sin(angle_rad) * yy
    pattern = torch.sin(2.0 * math.pi * cycles * direction)
    return pattern[None, None]


def _orientation_distance(a, b):
    return torch.atan2(torch.sin(2.0 * (a - b)), torch.cos(2.0 * (a - b))).abs() / 2.0


def test_horizontal_stripes_have_stable_orientation():
    learner = FourierOrientationLearner(
        patch_size=17,
        num_angle_bins=36,
        harmonic_orders=(2, 4, 6),
        detach_orientation=False,
    )
    feat = _oriented_sinusoid(0.0)

    out = learner(feat)

    center_theta = out["theta"][0, 16, 16]
    assert _orientation_distance(center_theta, torch.tensor(0.0)) < 0.25
    assert out["confidence"][0, 16, 16] > 0.05


def test_rotated_stripes_change_orientation():
    learner = FourierOrientationLearner(
        patch_size=17,
        num_angle_bins=72,
        harmonic_orders=(2, 4, 6),
        detach_orientation=False,
    )
    out_a = learner(_oriented_sinusoid(0.0))
    out_b = learner(_oriented_sinusoid(math.pi / 4.0))

    theta_a = out_a["theta"][0, 16, 16]
    theta_b = out_b["theta"][0, 16, 16]
    assert _orientation_distance(theta_a, theta_b) > 0.35


def test_constant_patch_has_low_confidence():
    learner = FourierOrientationLearner(patch_size=7, num_angle_bins=36)
    feat = torch.ones(1, 3, 15, 15)

    out = learner(feat)

    assert out["confidence"].max() < 0.05


def test_outputs_are_finite_and_fourier_code_shape_is_correct():
    learner = FourierOrientationLearner(
        patch_size=7,
        num_angle_bins=36,
        harmonic_orders=(2, 4, 6, 8),
    )
    feat = torch.randn(2, 5, 11, 13)

    out = learner(feat)

    assert out["theta"].shape == (2, 11, 13)
    assert out["confidence"].shape == (2, 11, 13)
    assert out["fourier_code"].shape == (2, 11, 13, 8)
    for value in out.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()

from argparse import Namespace

import numpy as np

from M_Tools.analysis.run_e9_scale_counterfactual_probe import (
    build_variants,
    resolve_scale_target_area,
)


def make_args(**overrides):
    data = {
        'small_area': 582.0,
        'scale_target_mode': 'pred_class',
        'pred_class_area_priors': {'ship': 650.0, 'large-vehicle': 1471.0},
        'scale_target_class': 'ship',
        'min_pred_class_scale': 0.10,
        'max_pred_class_scale': 2.50,
        'min_small_scale': 0.18,
        'max_small_scale': 0.55,
        'canvas_size': 1024,
        'crop_padding': 2,
    }
    data.update(overrides)
    return Namespace(**data)


def test_resolve_scale_target_area_prefers_pred_class_prior():
    area, label = resolve_scale_target_area(make_args(), fallback_class='ship')

    assert area == 650.0
    assert label == 'ship'


def test_build_variants_uses_pred_class_scale_name_and_area():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    qbox = np.array([[10, 10], [50, 10], [50, 50], [10, 50]],
                    dtype=np.float32)

    variants = build_variants(img, qbox, make_args(), scale_class='ship')

    assert [v['variant'] for v in variants] == [
        'original_tile',
        'neutral_same_scale',
        'neutral_pred_class_scale',
    ]
    assert abs(variants[-1]['target_area'] - 650.0) < 5.0
    assert 'ship' in variants[-1]['note']

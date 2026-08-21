from types import SimpleNamespace

import pytest
import torch

from M_Tools.analysis.fit_g2_adapter_from_predictions import (
    add_fit_row_rank_fields,
    rank_preserve_loss,
    rows_to_tensors,
)


def test_add_fit_row_rank_fields_assigns_global_image_and_class_rank():
    rows = [
        {
            "img_id": "img_a",
            "pred_label": 0,
            "score": 0.80,
            "target_good": 1.0,
        },
        {
            "img_id": "img_a",
            "pred_label": 0,
            "score": 0.95,
            "target_good": 0.0,
        },
        {
            "img_id": "img_b",
            "pred_label": 1,
            "score": 0.90,
            "target_good": 1.0,
        },
    ]

    ranked = add_fit_row_rank_fields(rows)

    by_key = {(row["img_id"], row["score"]): row for row in ranked}
    assert by_key[("img_a", 0.95)]["global_rank"] == 1
    assert by_key[("img_b", 0.90)]["global_rank"] == 2
    assert by_key[("img_a", 0.80)]["global_rank"] == 3
    assert by_key[("img_a", 0.95)]["image_rank"] == 1
    assert by_key[("img_a", 0.80)]["image_rank"] == 2
    assert by_key[("img_a", 0.95)]["class_rank"] == 1
    assert by_key[("img_a", 0.80)]["class_rank"] == 2


def test_rank_preserve_loss_only_penalizes_protected_good_rows():
    rows = [
        {
            "score": 0.90,
            "target_good": 1.0,
            "abs_z": 5.0,
            "signed_z": 5.0,
            "gaussian_log_prob": -13.0,
            "log_std": 0.0,
            "image_rank": 1,
            "class_rank": 1,
            "global_rank": 1,
        },
        {
            "score": 0.92,
            "target_good": 0.0,
            "abs_z": 5.0,
            "signed_z": 5.0,
            "gaussian_log_prob": -13.0,
            "log_std": 0.0,
            "image_rank": 1,
            "class_rank": 1,
            "global_rank": 1,
        },
        {
            "score": 0.20,
            "target_good": 1.0,
            "abs_z": 5.0,
            "signed_z": 5.0,
            "gaussian_log_prob": -13.0,
            "log_std": 0.0,
            "image_rank": 100,
            "class_rank": 100,
            "global_rank": 100,
        },
    ]
    data = rows_to_tensors(rows, torch.device("cpu"))
    args = SimpleNamespace(
        rank_preserve_weight=10.0,
        rank_preserve_margin=0.05,
        rank_protect_score_thr=0.5,
        rank_protect_image_rank=10,
        rank_protect_class_rank=10,
        rank_protect_global_rank=0,
    )

    loss = rank_preserve_loss(
        delta=torch.tensor([-0.25, -0.25, -0.25]),
        data=data,
        args=args,
    )
    protected_only_loss = 10.0 * ((0.25 - 0.05) ** 2)

    assert loss.item() == pytest.approx(protected_only_loss)


def test_rank_preserve_loss_is_zero_when_protected_delta_is_within_margin():
    rows = [
        {
            "score": 0.90,
            "target_good": 1.0,
            "abs_z": 5.0,
            "signed_z": 5.0,
            "gaussian_log_prob": -13.0,
            "log_std": 0.0,
            "image_rank": 1,
            "class_rank": 1,
            "global_rank": 1,
        },
    ]
    data = rows_to_tensors(rows, torch.device("cpu"))
    args = SimpleNamespace(
        rank_preserve_weight=10.0,
        rank_preserve_margin=0.05,
        rank_protect_score_thr=0.5,
        rank_protect_image_rank=10,
        rank_protect_class_rank=10,
        rank_protect_global_rank=0,
    )

    loss = rank_preserve_loss(
        delta=torch.tensor([-0.03]),
        data=data,
        args=args,
    )

    assert loss.item() == pytest.approx(0.0)

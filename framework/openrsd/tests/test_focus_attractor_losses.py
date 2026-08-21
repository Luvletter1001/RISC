import torch

from M_AD.models.losses.focus_attractor_losses import (
    anti_attractor_loss,
    migration_kl_loss,
    preserve_loss,
    support_distill_loss,
)


def test_support_distill_loss_is_zero_for_identical_supports():
    support = torch.randn(2, 3, 4, 5)

    loss = support_distill_loss(support, support)

    assert loss.item() == 0.0


def test_anti_attractor_loss_uses_only_negative_mask():
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 1] = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    negative_mask = torch.tensor([[[True, False], [False, True]]])

    masked = anti_attractor_loss(logits, sv_class_index=1,
                                 negative_mask=negative_mask, margin=0.0)
    empty = anti_attractor_loss(logits, sv_class_index=1,
                                negative_mask=torch.zeros_like(negative_mask),
                                margin=0.0)

    assert masked > 0
    assert empty.item() == 0.0


def test_preserve_loss_prefers_small_vehicle_on_positive_mask():
    logits = torch.zeros(1, 3, 1, 2)
    logits[:, 1, :, 0] = 4.0
    logits[:, 0, :, 1] = 4.0
    positive_mask = torch.tensor([[[True, False]]])

    loss = preserve_loss(logits, sv_class_index=1, positive_mask=positive_mask)

    assert loss < 0.05


def test_migration_kl_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 4, 3, 3)

    loss = migration_kl_loss(logits, logits)

    assert loss.item() == 0.0

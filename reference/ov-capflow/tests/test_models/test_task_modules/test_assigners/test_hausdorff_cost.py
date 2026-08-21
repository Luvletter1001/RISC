import math

import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS


def _instances(boxes):
    return InstanceData(bboxes=torch.tensor(boxes, dtype=torch.float32))


def test_hausdorff_cost_is_public_and_registry_buildable():
    from mmrotate.models.task_modules.assigners import HausdorffCost

    cost = TASK_UTILS.build(dict(type='HausdorffCost', weight=2.0))
    assert isinstance(cost, HausdorffCost)
    assert cost.weight == 2.0


def test_hausdorff_cost_is_symmetric_and_pi_periodic():
    from mmrotate.models.task_modules.assigners import HausdorffCost

    cost = HausdorffCost(weight=1.0, num_points=4)
    first = _instances([
        [50, 40, 20, 10, 0.0],
        [25, 20, 8, 16, 0.3],
    ])
    second = _instances([
        [50, 40, 20, 10, math.pi],
        [70, 60, 12, 18, -0.2],
    ])
    meta = dict(img_shape=(100, 100))

    forward = cost(first, second, meta)
    backward = cost(second, first, meta)

    assert forward.shape == (2, 2)
    assert torch.allclose(forward, backward.T, atol=1e-6)
    assert forward[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_hausdorff_cost_handles_empty_gt_and_has_finite_gradient():
    from mmrotate.models.task_modules.assigners import HausdorffCost

    cost = HausdorffCost(weight=1.0, num_points=8)
    pred_boxes = torch.tensor(
        [[50, 40, 20, 10, 0.1], [25, 20, 8, 16, 0.3]],
        dtype=torch.float32,
        requires_grad=True)
    pred = InstanceData(bboxes=pred_boxes)
    empty = InstanceData(bboxes=torch.empty(0, 5))
    meta = dict(img_shape=(100, 100))

    empty_cost = cost(pred, empty, meta)
    assert empty_cost.shape == (2, 0)

    target = _instances([[55, 42, 18, 12, -0.2]])
    loss = cost(pred, target, meta).sum()
    loss.backward()
    assert pred_boxes.grad is not None
    assert torch.isfinite(pred_boxes.grad).all()


def test_hausdorff_cost_dense_gt_fallback_preserves_shape():
    from mmrotate.models.task_modules.assigners import HausdorffCost

    cost = HausdorffCost(weight=1.0, num_points=8)
    pred = _instances([[50, 40, 20, 10, 0.1]])
    dense_boxes = torch.tensor(
        [[float(i % 100), float(i // 100), 4.0, 2.0, 0.0]
         for i in range(700)])
    dense = InstanceData(bboxes=dense_boxes)
    result = cost(pred, dense, dict(img_shape=(100, 100)))

    assert result.shape == (1, 700)
    assert torch.isfinite(result).all()

from __future__ import annotations

import math
import unittest

import torch

from mmrotate.models.dense_heads.rotated_reppoints_head import _points_to_rbox_cpu


class OrientedRepPointsSafePostprocessTest(unittest.TestCase):
    def test_converts_pointset_to_finite_rbox_on_cpu_safe_path(self):
        pointsets = torch.tensor([[
            10.0, 10.0, 20.0, 10.0, 20.0, 20.0,
            10.0, 20.0, 15.0, 15.0, 18.0, 12.0,
            12.0, 18.0, 11.0, 14.0, 19.0, 16.0,
        ]])

        rboxes = _points_to_rbox_cpu(pointsets)

        self.assertEqual(tuple(rboxes.shape), (1, 5))
        self.assertTrue(torch.isfinite(rboxes).all().item())
        self.assertGreaterEqual(float(rboxes[0, 2]), 1.0)
        self.assertGreaterEqual(float(rboxes[0, 3]), 1.0)
        self.assertLessEqual(abs(float(rboxes[0, 4])), math.pi)


if __name__ == "__main__":
    unittest.main()

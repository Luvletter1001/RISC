#!/usr/bin/env python3
"""Unit tests for clean SV-vs-GT conflict visualization filtering."""
from __future__ import annotations

import unittest

from M_Tools.analysis.draw_sv_gt_conflict_failures import is_sv_gt_conflict_event
from M_Tools.analysis.draw_sv_gt_conflict_failures import is_strict_sv_misclassification


class SvGtConflictFilterTest(unittest.TestCase):
    def test_keeps_only_case1_non_sv_non_large_vehicle_gt(self) -> None:
        self.assertTrue(is_sv_gt_conflict_event({"event": "sv_case1", "gt_class": "ship"}))
        self.assertTrue(is_sv_gt_conflict_event({"event": "sv_case1", "gt_class": "storage_tank"}))

        self.assertFalse(is_sv_gt_conflict_event({"event": "sv_bg", "gt_class": ""}))
        self.assertFalse(is_sv_gt_conflict_event({"event": "sv_residual", "gt_class": "small-vehicle"}))
        self.assertFalse(is_sv_gt_conflict_event({"event": "sv_case1", "gt_class": "large-vehicle"}))
        self.assertFalse(is_sv_gt_conflict_event({"event": "sv_case1", "gt_class": "background"}))
        self.assertFalse(is_sv_gt_conflict_event({"event": "sv_case1", "gt_class": "small_vehicle"}))

    def test_strict_misclassification_requires_sv_pred_eligible_gt_and_high_iou(self) -> None:
        self.assertTrue(is_strict_sv_misclassification("Small_Vehicle", "ship", 0.72, 0.70))

        self.assertFalse(is_strict_sv_misclassification("Ship", "small-vehicle", 0.91, 0.70))
        self.assertFalse(is_strict_sv_misclassification("Small_Vehicle", "small-vehicle", 0.91, 0.70))
        self.assertFalse(is_strict_sv_misclassification("Small_Vehicle", "large-vehicle", 0.91, 0.70))
        self.assertFalse(is_strict_sv_misclassification("Small_Vehicle", "ship", 0.69, 0.70))


if __name__ == "__main__":
    unittest.main()

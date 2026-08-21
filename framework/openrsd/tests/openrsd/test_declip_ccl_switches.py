from pathlib import Path
import unittest

import numpy as np
import torch
from mmengine import ConfigDict

from M_AD.models.detectors.Flex_Rtmdet_v3_1_formal import (
    sample_paired_support_rows,
    select_active_support_feat_dict,
)
from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import compute_ccl_loss
from tools.openrsd_config_switches import sync_openrsd_feature_switches


class DeclipCclSwitchTest(unittest.TestCase):

    def test_select_active_support_feat_dict_uses_baseline_when_declip_disabled(
        self,
    ):
        tmp_path = Path('/tmp/openrsd_switch_test')
        baseline = {'Data1_DOTA2': str(tmp_path / 'baseline.pkl')}
        declip = {'Data1_DOTA2': str(tmp_path / 'declip.pkl')}

        selected = select_active_support_feat_dict(
            support_feat_dict=baseline,
            use_declip_support=False,
            declip_support_feat_dict=declip,
        )

        self.assertIs(selected, baseline)

    def test_select_active_support_feat_dict_requires_declip_dict_when_enabled(
        self,
    ):
        with self.assertRaisesRegex(ValueError, 'declip_support_feat_dict'):
            select_active_support_feat_dict(
                support_feat_dict={'Data1_DOTA2': 'baseline.pkl'},
                use_declip_support=True,
                declip_support_feat_dict=None,
            )

    def test_compute_ccl_loss_returns_finite_scalar_for_positive_assignments(
        self,
    ):
        dense_embeds = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
        assigned_labels = torch.tensor([[0, 0, 1]])
        support_feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        support_labels = torch.tensor([[0, 1]])

        loss = compute_ccl_loss(
            dense_embeds=dense_embeds,
            assigned_labels=assigned_labels,
            support_feats=support_feats,
            support_labels=support_labels,
            num_classes=2,
            temperature=0.1,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sync_openrsd_feature_switches_propagates_top_level_cfg_options(
        self,
    ):
        cfg = ConfigDict(
            use_declip=True,
            use_ccl=True,
            model=ConfigDict(
                use_declip_support=False,
                bbox_head=ConfigDict(use_ccl_loss=False),
            ),
        )

        sync_openrsd_feature_switches(cfg)

        self.assertTrue(cfg.model.use_declip_support)
        self.assertTrue(cfg.model.bbox_head.use_ccl_loss)

    def test_sample_paired_support_rows_uses_shortest_feature_bank(self):
        text_feats = np.zeros((80, 768), dtype=np.float32)
        visual_feats = np.zeros((50, 1024), dtype=np.float32)
        labels = np.zeros((80,), dtype=np.int64)

        sampled_text, sampled_visual, sampled_labels = sample_paired_support_rows(
            text_feats,
            visual_feats,
            labels,
            support_shot=6,
        )

        self.assertEqual(sampled_text.shape, (6, 768))
        self.assertEqual(sampled_visual.shape, (6, 1024))
        self.assertEqual(sampled_labels.shape, (6,))


if __name__ == '__main__':
    unittest.main()

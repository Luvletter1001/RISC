# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.registry import MODELS as MMDET_MODELS

from mmrotate.registry import MODELS as MMROTATE_MODELS


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class StripRCNN(TwoStageDetector):
    """Strip R-CNN detector.

    This is a thin MMRotate 1.x wrapper around the two-stage detector stack so
    legacy Strip R-CNN configs can keep using ``type='StripRCNN'``.
    """

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        backbone_prefix = prefix + 'backbone.' if prefix else 'backbone.'
        has_backbone_prefix = any(
            key.startswith(backbone_prefix) for key in state_dict)
        has_unprefixed_backbone = any(
            key.startswith(('patch_embed', 'block', 'norm'))
            for key in state_dict)
        if not has_backbone_prefix and has_unprefixed_backbone:
            for key in list(state_dict.keys()):
                if key.startswith(('patch_embed', 'block', 'norm')):
                    state_dict[backbone_prefix + key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward_dummy(self, img):
        outs = ()
        x = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

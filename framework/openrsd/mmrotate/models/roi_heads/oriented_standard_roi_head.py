# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.registry import MODELS as MMDET_MODELS

from mmrotate.registry import MODELS as MMROTATE_MODELS


@MMDET_MODELS.register_module()
@MMROTATE_MODELS.register_module()
class OrientedStandardRoIHead(StandardRoIHead):
    """Compatibility alias for legacy MMRotate Strip R-CNN configs."""

    pass

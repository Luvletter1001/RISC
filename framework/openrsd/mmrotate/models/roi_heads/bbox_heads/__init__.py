# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import RotatedShared2FCBBoxHead
from .gv_bbox_head import GVBBoxHead
from .strip_head import StripHead, StripHead_

__all__ = ['RotatedShared2FCBBoxHead', 'GVBBoxHead', 'StripHead', 'StripHead_']

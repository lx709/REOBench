# Copyright (c) OpenMMLab. All rights reserved.
from .re_fpn import ReFPN
# ========== 迁移自mmdetection的necks ==========
from .frequency_spatial_fpn import FrequencySpatialFPN  # 注释：需要确认与mmrotate兼容性

__all__ = ['ReFPN']
__all__ += ['FrequencySpatialFPN']  # 注释：暂时未激活

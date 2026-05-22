# Copyright (c) OpenMMLab. All rights reserved.
from .re_resnet import ReResNet
# ========== 迁移自mmdetection的backbones ==========
from .msfa import MSFA  # 注释：需要确认与mmrotate兼容性
# from .convnext_moe import ConvNeXt_moe, ConvNeXt_moe_MultiInput  # 注释：需要mmdetection依赖
from .hivit import HiViT
from .FFTresnet import FFTResNet
from .sarclip_vit import SARCLIPViT
from .sarmae_vit import SARMAEViT
from .sarmae_vit_timm import VisionTransformer_timm
from .sarwmixmae_mixmim import SARWMixMIM

__all__ = ['ReResNet',
'MSFA',
'HiViT',
'FFTResNet',
'SARCLIPViT',
'SARMAEViT',
'SARWMixMIM',
'VisionTransformer_timm',

]
# __all__ += ['MSFA', 'ConvNeXt_moe', 'ConvNeXt_moe_MultiInput', 'HiViT']  # 注释：暂时未激活

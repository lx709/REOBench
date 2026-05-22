# # Copyright (c) OpenMMLab. All rights reserved.
# """SMARTIES Simple Segmentation Head.

# This is a minimal segmentation head that mimics SMARTIES' approach:
# - Single 1x1 conv to map features to num_classes
# - Bilinear upsampling to target size

# Reference: SMARTIES utils/utils.py:148-152
#     torch.nn.Sequential(
#         torch.nn.Conv2d(model.head.in_features, nb_classes, kernel_size=1),
#         torch.nn.Upsample(scale_factor=96/14, mode='bilinear', align_corners=False)
#     )
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from mmseg.registry import MODELS
# from .decode_head import BaseDecodeHead


# @MODELS.register_module()
# class SMARTIESSimpleHead(BaseDecodeHead):
#     """SMARTIES-style Simple Segmentation Head.

#     This head implements the same architecture as SMARTIES DFC2020:
#     1. Select last transformer feature (from all_tokens output)
#     2. Apply 1x1 Conv: in_channels -> num_classes
#     3. Bilinear Upsample: feature_size -> target_size

#     Architecture:
#         Input: Backbone features list from out_indices=[3,5,7,11]
#         Select: Last feature (index -1 by default)
#         Conv2d(in_channels=768, num_classes, kernel_size=1, no bias term in SMARTIES)
#         Upsample(mode='bilinear', align_corners=False)
#         Output: (B, num_classes, target_H, target_W)

#     Args:
#         in_channels (int): Number of input channels. For ViT-Base, this is 768.
#         num_classes (int): Number of segmentation classes.
#         in_index (int): Which feature from backbone output list to use.
#             Default: -1 (last feature, which is from the last transformer block).
#         target_size (int or tuple): Target output size. For DFC2020, this is 96.
#             Default: 96.
#         interpolate_mode (str): Upsampling mode. Default: 'bilinear'.
#         align_corners (bool): Whether to align corners in interpolation.
#             Default: False (same as SMARTIES).

#     Example:
#         For DFC2020 with ViT-Base/16 on 96x96 images:
#         - Backbone outputs: list of 4 features, each (B, 768, 6, 6)  # 96/16=6
#         - Select last: (B, 768, 6, 6)
#         - After Conv2d: (B, 8, 6, 6)
#         - After Upsample: (B, 8, 96, 96)

#         Scale factor calculation: 96 / 6 = 16.0
#         But SMARTIES uses 96/14 ≈ 6.857 because they count patches differently
#         (probably 224/16 = 14 patches from pretraining, then finetuned to 96x96)
#     """

#     def __init__(self,
#                  target_size=96,
#                  interpolate_mode='bilinear',
#                  align_corners=False,
#                  **kwargs):
#         # Set input_transform to 'resize_concat' or 'multiple_select' to handle list input
#         # Actually, we'll use default which calls _transform_inputs
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.target_size = target_size if isinstance(target_size, (list, tuple)) else (target_size, target_size)
#         self.interpolate_mode = interpolate_mode
#         self.align_corners = align_corners

#         # SMARTIES uses a simple 1x1 conv (no BN, no activation, no bias)
#         # The base class creates self.conv_seg, but we want to ensure no bias
#         self.conv_seg = nn.Conv2d(
#             self.in_channels,
#             self.num_classes,
#             kernel_size=1,
#             bias=False  # SMARTIES Conv2d typically has bias=True by default, but let's keep it simple
#         )

#     def forward(self, inputs):
#         """Forward function.

#         Args:
#             inputs (list[Tensor]): List of multi-level feature maps from backbone.
#                 For SMARTIES backbone with out_indices=[3,5,7,11]:
#                     - inputs[0]: features from block 3, shape (B, 768, 6, 6)
#                     - inputs[1]: features from block 5, shape (B, 768, 6, 6)
#                     - inputs[2]: features from block 7, shape (B, 768, 6, 6)
#                     - inputs[3]: features from block 11, shape (B, 768, 6, 6)

#         Returns:
#             Tensor: Segmentation logits of shape (B, num_classes, target_H, target_W).
#         """
#         # _transform_inputs will select features based on in_index
#         # With input_transform='multiple_select' and in_index=-1 (or 3),
#         # it selects the last feature
#         x = self._transform_inputs(inputs)  # (B, 768, 6, 6)

#         # Apply 1x1 conv: (B, in_channels, H', W') -> (B, num_classes, H', W')
#         output = self.conv_seg(x)  # (B, 8, 6, 6)

#         # Upsample to target size: (B, num_classes, H', W') -> (B, num_classes, target_H, target_W)
#         output = F.interpolate(
#             output,
#             size=self.target_size,
#             mode=self.interpolate_mode,
#             align_corners=self.align_corners if self.interpolate_mode != 'nearest' else None
#         )  # (B, 8, 96, 96)

#         return output
# Copyright (c) OpenMMLab. All rights reserved.
"""SMARTIES Simple Segmentation Head.

This is a minimal segmentation head that mimics SMARTIES' approach:
- Single 1x1 conv to map features to num_classes
- Bilinear upsampling to target size

Reference: SMARTIES utils/utils.py:148-152
    torch.nn.Sequential(
        torch.nn.Conv2d(model.head.in_features, nb_classes, kernel_size=1),
        torch.nn.Upsample(scale_factor=96/14, mode='bilinear', align_corners=False)
    )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class SMARTIESSimpleHead(BaseDecodeHead):
    """SMARTIES-style Simple Segmentation Head.

    This head implements the same architecture as SMARTIES DFC2020:
    1. Select last transformer feature (from all_tokens output)
    2. Apply 1x1 Conv: in_channels -> num_classes
    3. Bilinear Upsample: feature_size -> target_size

    Architecture:
        Input: Backbone features list from out_indices=[3,5,7,11]
        Select: Last feature (index -1 by default)
        Conv2d(in_channels=768, num_classes, kernel_size=1, no bias term in SMARTIES)
        Upsample(mode='bilinear', align_corners=False)
        Output: (B, num_classes, target_H, target_W)

    Args:
        in_channels (int): Number of input channels. For ViT-Base, this is 768.
        num_classes (int): Number of segmentation classes.
        in_index (int): Which feature from backbone output list to use.
            Default: -1 (last feature, which is from the last transformer block).
        target_size (int or tuple): Target output size. For DFC2020, this is 96.
            Default: 96.
        interpolate_mode (str): Upsampling mode. Default: 'bilinear'.
        align_corners (bool): Whether to align corners in interpolation.
            Default: False (same as SMARTIES).

    Example:
        For DFC2020 with ViT-Base/16 on 96x96 images:
        - Backbone outputs: list of 4 features, each (B, 768, 6, 6)  # 96/16=6
        - Select last: (B, 768, 6, 6)
        - After Conv2d: (B, 8, 6, 6)
        - After Upsample: (B, 8, 96, 96)

        Scale factor calculation: 96 / 6 = 16.0
        But SMARTIES uses 96/14 ≈ 6.857 because they count patches differently
        (probably 224/16 = 14 patches from pretraining, then finetuned to 96x96)
    """

    def __init__(self,
                 target_size=96,
                 interpolate_mode='bilinear',
                 align_corners=False,
                 **kwargs):
        # Support both single feature (in_index=int) and multiple features (in_index=list)
        # When in_index is list, use 'multiple_select' transform
        # When in_index is int, use no transform (default)
        if 'in_index' in kwargs and isinstance(kwargs['in_index'], (list, tuple)):
            # Multiple features - need input_transform
            super().__init__(input_transform='multiple_select', **kwargs)
            # When using multiple_select, in_channels is a list
            # Concatenate all features, so total channels = sum
            fused_channels = sum(self.in_channels) if isinstance(self.in_channels, (list, tuple)) else self.in_channels
        else:
            # Single feature - no transform needed
            super().__init__(**kwargs)
            fused_channels = self.in_channels

        self.target_size = target_size if isinstance(target_size, (list, tuple)) else (target_size, target_size)
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        # SMARTIES uses a simple 1x1 conv (no BN, no activation)
        self.conv_seg = nn.Conv2d(
            fused_channels,
            self.num_classes,
            kernel_size=1,
            bias=True
        )

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): List of multi-level feature maps from backbone.
                For SMARTIES backbone with out_indices=[3,5,7,11]:
                    - inputs[0]: features from block 3, shape (B, 768, 6, 6)
                    - inputs[1]: features from block 5, shape (B, 768, 6, 6)
                    - inputs[2]: features from block 7, shape (B, 768, 6, 6)
                    - inputs[3]: features from block 11, shape (B, 768, 6, 6)

        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, target_H, target_W).
        """
        # _transform_inputs handles both cases:
        # - Single feature (in_index=int): returns inputs[in_index]
        # - Multiple features (in_index=list): returns list of selected features
        x = self._transform_inputs(inputs)

        # If multiple features, concatenate them
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # Concat along channel dim

        # Apply 1x1 conv: (B, in_channels, H', W') -> (B, num_classes, H', W')
        output = self.conv_seg(x)

        # Upsample to target size: (B, num_classes, H', W') -> (B, num_classes, target_H, target_W)
        output = F.interpolate(
            output,
            size=self.target_size,
            mode=self.interpolate_mode,
            align_corners=self.align_corners if self.interpolate_mode != 'nearest' else None
        )  # (B, 8, 96, 96)

        return output

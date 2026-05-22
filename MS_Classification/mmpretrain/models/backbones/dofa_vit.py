# Copyright (c) OpenMMLab. All rights reserved.
"""DOFA (Dynamic One-For-All) Vision Transformer Backbone.

This is a self-contained implementation adapted for mmpretrain.
"""
from functools import partial
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import load_checkpoint

from mmpretrain.registry import MODELS


# ============================================================================
# Position Embedding Utilities
# ============================================================================
def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings (torch version)."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ============================================================================
# Dynamic Weight Generator (Transformer-based)
# ============================================================================
class TransformerWeightGenerator(nn.Module):
    """Transformer-based dynamic weight generator for wavelength-adaptive convolution."""

    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num: -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


# ============================================================================
# FC Residual Layer
# ============================================================================
class FCResLayer(nn.Module):
    """Fully connected residual layer."""

    def __init__(self, linear_size=128):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


# ============================================================================
# Dynamic MLP for One-For-All Patch Embedding
# ============================================================================
class Dynamic_MLP_OFA(nn.Module):
    """Dynamic MLP for wavelength-adaptive patch embedding."""

    def __init__(self, wv_planes, inter_dim=128, kernel_size=16, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)
        return dynamic_weights

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


# ============================================================================
# DOFA Vision Transformer
# ============================================================================
class OFAViT(nn.Module):
    """One-For-All Vision Transformer with dynamic wavelength adaptation."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim
        )

        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x, wave_list):
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        fx = self.forward_features(x, wave_list)
        x = self.forward_head(fx)
        return x, fx


# ============================================================================
# MMPretrain-compatible DOFA Backbone
# ============================================================================
@MODELS.register_module()
class DOFAViT(BaseModule):
    """DOFA Vision Transformer Backbone for MMPretrain.

    Args:
        img_size (int): Input image size. Default: 120.
        patch_size (int): Patch size. Default: 16.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        wv_planes (int): Wavelength encoding dimension. Default: 128.
        global_pool (bool): Use global average pooling. Default: True.
        drop_rate (float): Dropout rate. Default: 0.0.
        wavelengths (list): Wavelengths for each channel (in micrometers).
        pretrained (str, optional): Path to pretrained weights.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        img_size=120,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        wv_planes=128,
        global_pool=True,
        drop_rate=0.0,
        wavelengths=None,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        # Default wavelengths for Sentinel-2 (12 bands, in micrometers)
        if wavelengths is None:
            # Sentinel-2 band wavelengths (B1-B12, excluding B10)
            wavelengths = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
                          0.783, 0.842, 0.865, 1.610, 2.190]

        self.wavelengths = wavelengths
        self.embed_dim = embed_dim

        self.model = OFAViT(
            img_size=img_size,
            patch_size=patch_size,
            drop_rate=drop_rate,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            wv_planes=wv_planes,
            num_classes=0,  # No head for backbone
            global_pool=global_pool,
            mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # Remove the classification head
        self.model.head = nn.Identity()

        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        if pretrained is not None:
            load_checkpoint(self, pretrained, strict=False, map_location='cpu')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple: Output features.
        """
        _, features = self.model(x, self.wavelengths)
        return (features,)


def dofa_vit_base_patch16(**kwargs):
    """DOFA ViT-Base model."""
    model = OFAViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def dofa_vit_large_patch16(**kwargs):
    """DOFA ViT-Large model."""
    model = OFAViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

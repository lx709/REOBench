# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 移植自 SARMAE_Fintune vit_timm.py，适配 mmrotate

from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmengine.dist import get_dist_info
from mmrotate.registry import MODELS
import math
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

@MODELS.register_module()
class VisionTransformer_timm(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, pretrained=None,
                 use_checkpoint=False, out_indices=[11], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        from timm.models.vision_transformer import Block
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                drop_path=dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        pretrained = self.pretrained
        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = self.patch_embed.grid_size
                num_patches = self.patch_embed.num_patches
                if 'cls_token' in state_dict.keys():
                    num_extra_tokens = 1
                else:
                    num_extra_tokens = 0
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                new_size = int(num_patches ** 0.5)
                if orig_size != new_size:
                    if rank == 0:
                        print(f"Position interpolate from {orig_size}x{orig_size} to {H}x{W}")
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]
            msg = self.load_state_dict(state_dict, False)
            if rank == 0:
                print(msg)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        Hp = Wp = int(self.img_size // self.patch_size)
        batch_size, seq_len, _ = x.size()
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        features = [self.fpn1(xp), self.fpn2(xp), self.fpn3(xp), self.fpn4(xp)]
        return tuple(features)

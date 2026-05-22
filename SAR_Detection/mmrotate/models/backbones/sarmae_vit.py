import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import Block

from mmrotate.registry import MODELS


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


@MODELS.register_module()
class SARMAEViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_path_rate=0.3,
                 use_checkpoint=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
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

        if patch_size != 16:
            raise ValueError(f'SARMAEViT only supports patch_size=16, got {patch_size}.')

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
        self._freeze_stages()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for block in self.blocks[:self.frozen_stages]:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def _resize_pos_embed(self, pos_embed, target_hw):
        if pos_embed.ndim != 3:
            return pos_embed
        num_tokens = pos_embed.shape[1]
        size = int(math.sqrt(num_tokens))
        target_h, target_w = target_hw
        if size * size != num_tokens:
            return pos_embed
        if size == target_h and size == target_w:
            return pos_embed
        pos_embed = pos_embed.reshape(1, size, size, -1).permute(0, 3, 1, 2).contiguous()
        pos_embed = nn.functional.interpolate(
            pos_embed, size=(target_h, target_w), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous().reshape(1, target_h * target_w, -1).contiguous()
        return pos_embed

    def _adapt_input_proj(self, weight):
        if weight.shape[1] == self.in_chans:
            return weight
        if weight.shape[1] == 1:
            return weight.repeat(1, self.in_chans, 1, 1) / self.in_chans
        if self.in_chans == 1:
            return weight.mean(dim=1, keepdim=True)
        if weight.shape[1] < self.in_chans:
            repeat = math.ceil(self.in_chans / weight.shape[1])
            weight = weight.repeat(1, repeat, 1, 1)[:, :self.in_chans]
            return weight * (weight.shape[1] / self.in_chans)
        return weight[:, :self.in_chans]

    def init_weights(self):
        checkpoint_path = self.pretrained
        if checkpoint_path is None and self.init_cfg is not None:
            checkpoint_path = self.init_cfg.get('checkpoint')
        if checkpoint_path is None:
            return

        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', checkpoint_data.get('model', checkpoint_data))

        # 只保留 sar_encoder. 前缀的参数，并去掉前缀
        sar_encoder_prefix = 'sar_encoder.'
        filtered = {k[len(sar_encoder_prefix):]: v for k, v in state_dict.items() if k.startswith(sar_encoder_prefix)}

        # 只保留和当前模型参数名一致的key
        model_keys = set(self.state_dict().keys())
        cleaned = {k: v for k, v in filtered.items() if k in model_keys}

        if 'patch_embed.proj.weight' in cleaned:
            cleaned['patch_embed.proj.weight'] = self._adapt_input_proj(cleaned['patch_embed.proj.weight'])

        # 处理pos_embed插值，严格去掉cls_token后再reshape/interpolate，兼容ViT/CLIP权重
        pos_embed = cleaned.pop('pos_embed', None)
        if pos_embed is not None:
            num_patches = self.patch_embed.num_patches
            embed_dim = self.embed_dim
            target_hw = self.patch_embed.grid_size
            # 先去掉cls_token（只保留patch部分）
            if pos_embed.shape[1] == 1 + 196:
                # 标准ViT/CLIP权重，14x14+1
                pos_tokens = pos_embed[:, 1:]
            elif pos_embed.shape[1] == num_patches + 1:
                # 罕见情况，权重patch数和模型一致
                pos_tokens = pos_embed[:, 1:]
            elif pos_embed.shape[1] == num_patches:
                pos_tokens = pos_embed
            else:
                # 自动推断patch部分
                if pos_embed.shape[1] > 1 and int((pos_embed.shape[1] - 1) ** 0.5) ** 2 == pos_embed.shape[1] - 1:
                    pos_tokens = pos_embed[:, 1:]
                elif int(pos_embed.shape[1] ** 0.5) ** 2 == pos_embed.shape[1]:
                    pos_tokens = pos_embed
                else:
                    raise RuntimeError(f"pos_embed patch tokens数量无法reshape为正方形，原始shape: {pos_embed.shape}")
            orig_size = int(pos_tokens.shape[1] ** 0.5)
            if orig_size * orig_size != pos_tokens.shape[1]:
                raise RuntimeError(f"去除cls_token后，patch tokens数量无法reshape为正方形，shape: {pos_tokens.shape}")
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=target_hw, mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)
            self.pos_embed.data.copy_(pos_tokens)

        cleaned.pop('head.weight', None)
        cleaned.pop('head.bias', None)
        result = self.load_state_dict(cleaned, strict=False)
        print(f'[SARMAEViT] loaded checkpoint from {checkpoint_path}')
        if result.missing_keys:
            print(f'[SARMAEViT] missing keys ({len(result.missing_keys)}): {result.missing_keys}')
        if result.unexpected_keys:
            print(f'[SARMAEViT] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}')

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = self.patch_embed(x)

        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        pos_embed = self._resize_pos_embed(self.pos_embed, (patch_h, patch_w))
        x = self.pos_drop(x + pos_embed)

        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().reshape(batch_size, self.embed_dim, patch_h, patch_w).contiguous()
        return (
            self.fpn1(x),
            self.fpn2(x),
            self.fpn3(x),
            self.fpn4(x),
        )
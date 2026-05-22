import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, PatchEmbed, DropPath, trunc_normal_, to_2tuple
from torch.utils.checkpoint import checkpoint

from mmrotate.registry import MODELS


def window_partition(x, window_size):
    batch_size, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(
        batch_size,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
        channels).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, channels).contiguous()
    return windows, (padded_h, padded_w)


def window_reverse(windows, window_size, padded_hw, hw):
    padded_h, padded_w = padded_hw
    height, width = hw
    batch_size = windows.shape[0] // (padded_h * padded_w // window_size // window_size)
    x = windows.view(
        batch_size,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, padded_h, padded_w, -1).contiguous()
    return x[:, :height, :width, :].contiguous()


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixMIMBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, hw):
        height, width = hw
        batch_size, _, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch_size, height, width, channels).contiguous()
        windows, padded_hw = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels).contiguous()
        windows = self.attn(windows)
        windows = windows.view(-1, self.window_size, self.window_size, channels).contiguous()
        x = window_reverse(windows, self.window_size, padded_hw, hw)
        x = x.view(batch_size, height * width, channels).contiguous()
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, hw):
        height, width = hw
        batch_size, _, channels = x.shape
        x = x.view(batch_size, height, width, channels).contiguous()
        if height % 2 == 1 or width % 2 == 1:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        next_hw = (x.shape[1], x.shape[2])
        x = x.view(batch_size, -1, 4 * channels).contiguous()
        x = self.norm(x)
        x = self.reduction(x)
        return x, next_hw


class MixMIMStage(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MixMIMBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, hw):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x, hw, use_reentrant=False)
            else:
                x = block(x, hw)
        stage_out = x
        stage_hw = hw
        if self.downsample is not None:
            x, hw = self.downsample(x, hw)
        return stage_out, stage_hw, x, hw


@MODELS.register_module()
class SARWMixMIM(nn.Module):
    def __init__(self,
                 img_size=800,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 window_size=(8, 8, 8, 4),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.out_norms = nn.ModuleList()
        start = 0
        for index, depth in enumerate(depths):
            dim = int(embed_dim * 2 ** index)
            self.stages.append(MixMIMStage(
                dim=dim,
                depth=depth,
                num_heads=num_heads[index],
                window_size=window_size[index],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[start:start + depth],
                downsample=index < len(depths) - 1,
                use_checkpoint=use_checkpoint))
            self.out_norms.append(nn.LayerNorm(dim, eps=1e-6))
            start += depth

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

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for stage in self.stages[:self.frozen_stages]:
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _resize_pos_embed(self, pos_embed, hw):
        num_tokens = pos_embed.shape[1]
        size = int(math.sqrt(num_tokens))
        target_h, target_w = hw
        if size * size != num_tokens:
            return pos_embed
        if size == target_h and size == target_w:
            return pos_embed
        pos_embed = pos_embed.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(target_h, target_w), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
        return pos_embed

    def _adapt_input_proj(self, weight):
        source_channels = weight.shape[1]
        if source_channels == self.in_chans:
            return weight
        if source_channels == 1:
            return weight.repeat(1, self.in_chans, 1, 1) / self.in_chans
        if self.in_chans == 1:
            return weight.mean(dim=1, keepdim=True)
        if source_channels < self.in_chans:
            repeat = math.ceil(self.in_chans / source_channels)
            expanded = weight.repeat(1, repeat, 1, 1)[:, :self.in_chans]
            return expanded * (source_channels / self.in_chans)
        return weight[:, :self.in_chans]

    def init_weights(self):
        checkpoint_path = self.pretrained
        if checkpoint_path is None and self.init_cfg is not None:
            checkpoint_path = self.init_cfg.get('checkpoint')
        if checkpoint_path is None:
            return

        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', checkpoint_data.get('model', checkpoint_data))

        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('backbone.'):
                key = key[len('backbone.'):]
            if key.startswith('encoder.'):
                key = key[len('encoder.'):]
            # 自动将layers.替换为stages.，以适配本实现
            if key.startswith('layers.'):
                key = 'stages.' + key[len('layers.'):]
            cleaned[key] = value

        if 'patch_embed.proj.weight' in cleaned:
            cleaned['patch_embed.proj.weight'] = self._adapt_input_proj(cleaned['patch_embed.proj.weight'])

        if 'absolute_pos_embed' in cleaned:
            cleaned['absolute_pos_embed'] = self._resize_pos_embed(
                cleaned['absolute_pos_embed'], self.patch_embed.grid_size)

        cleaned.pop('head.weight', None)
        cleaned.pop('head.bias', None)
        result = self.load_state_dict(cleaned, strict=False)
        print(f'[SARWMixMAE] loaded checkpoint from {checkpoint_path}')
        if result.missing_keys:
            print(f'[SARWMixMAE] missing keys ({len(result.missing_keys)}): {result.missing_keys}')
        if result.unexpected_keys:
            print(f'[SARWMixMAE] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}')

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = self.patch_embed(x)
        hw = (height // self.patch_size, width // self.patch_size)

        pos_embed = self._resize_pos_embed(self.absolute_pos_embed, hw)
        x = self.pos_drop(x + pos_embed)

        outputs = []
        for stage, out_norm in zip(self.stages, self.out_norms):
            stage_out, stage_hw, x, hw = stage(x, hw)
            stage_out = out_norm(stage_out)
            stage_out = stage_out.transpose(1, 2).reshape(batch_size, -1, stage_hw[0], stage_hw[1]).contiguous()
            outputs.append(stage_out)
        return tuple(outputs)
import math
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from mmengine.logging import MMLogger
from mmengine.model import BaseModule

from mmrotate.registry import MODELS


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            ('c_proj', nn.Linear(mlp_width, d_model)),
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x + self.ls_1(self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0])
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio=mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, out_indices=None):
        features = []
        for index, block in enumerate(self.resblocks):
            x = block(x)
            if out_indices is not None and index in out_indices:
                features.append(x)
        return x, features


@MODELS.register_module()
class SARCLIPViT(BaseModule):
    def __init__(
        self,
        image_size=224,
        patch_size=32,
        in_channels=3,
        embed_dim=768,
        layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_indices=(2, 5, 8, 11),
        with_fpn=True,
        frozen_stages=-1,
        ls_init_value=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.layers = layers
        self.out_indices = tuple(out_indices)
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages

        grid_size = image_size // patch_size
        scale = embed_dim ** -0.5

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid_size * grid_size + 1, embed_dim))
        self.ln_pre = LayerNorm(embed_dim)
        self.transformer = Transformer(
            width=embed_dim,
            layers=layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
        )

        if self.with_fpn:
            assert len(self.out_indices) == 4, 'with_fpn=True expects exactly 4 out_indices.'
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
            )
            self.fpn3 = LayerNorm2d(embed_dim)
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
            )

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return

        for param in self.conv1.parameters():
            param.requires_grad = False
        self.class_embedding.requires_grad = False
        self.positional_embedding.requires_grad = False
        for param in self.ln_pre.parameters():
            param.requires_grad = False

        for index in range(min(self.frozen_stages, len(self.transformer.resblocks))):
            block = self.transformer.resblocks[index]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def _interpolate_pos_embed(self, height, width, dtype, device):
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size

        cls_pos = self.positional_embedding[:1]
        patch_pos = self.positional_embedding[1:]

        base_size = int(math.sqrt(patch_pos.shape[0]))
        patch_pos = patch_pos.reshape(1, base_size, base_size, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(patch_h, patch_w),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, self.embed_dim)
        cls_pos = cls_pos.unsqueeze(0)
        pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
        return pos_embed.to(device=device, dtype=dtype)

    def init_weights(self):
        # Hide init_cfg before calling super() so that mmengine's PretrainedInit
        # does not attempt torch.load() on our .safetensors file, which would
        # raise an UnpicklingError.  We then handle checkpoint loading ourselves.
        saved_cfg = self.init_cfg
        self.init_cfg = None
        super().init_weights()
        self.init_cfg = saved_cfg

        if saved_cfg is None:
            return

        checkpoint = saved_cfg.get('checkpoint')
        if not checkpoint:
            return

        if not os.path.isfile(checkpoint):
            raise ValueError(f'checkpoint path {checkpoint} is invalid')

        if checkpoint.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError('Loading .safetensors checkpoints requires safetensors.') from exc
            state_dict = load_file(checkpoint, device='cpu')
        else:
            loaded = torch.load(checkpoint, map_location='cpu')
            state_dict = loaded.get('state_dict', loaded)

        # If checkpoint contains a 'visual.' namespace (i.e. a full CLIP model),
        # only extract those keys so that text-encoder weights (same key names,
        # different widths) do not overwrite the visual tower weights.
        has_visual_prefix = any(
            k.startswith('visual.') or k.startswith('module.visual.')
            for k in state_dict
        )
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.visual.'):
                new_state_dict[key[len('module.visual.'):]] = value
            elif key.startswith('visual.'):
                new_state_dict[key[len('visual.'):]] = value
            elif not has_visual_prefix:
                # standalone visual checkpoint: handle module. wrapper if present
                if key.startswith('module.'):
                    new_state_dict[key[len('module.'):]] = value
                else:
                    new_state_dict[key] = value
            # else: skip keys that belong to the text encoder

        # 处理positional_embedding插值
        ckpt_pos_embed = new_state_dict.pop('positional_embedding', None)
        if ckpt_pos_embed is not None:
            model_pos_embed = self.positional_embedding
            embed_dim = model_pos_embed.shape[1]
            num_patches = model_pos_embed.shape[0] - 1
            # 判断是否有cls_token
            if ckpt_pos_embed.shape[0] == 1 + int((ckpt_pos_embed.shape[0] - 1) ** 0.5) ** 2:
                cls_token = ckpt_pos_embed[:1]
                patch_pos = ckpt_pos_embed[1:]
            else:
                cls_token = None
                patch_pos = ckpt_pos_embed
            orig_size = int(patch_pos.shape[0] ** 0.5)
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
            target_size = int(num_patches ** 0.5)
            patch_pos = torch.nn.functional.interpolate(
                patch_pos, size=(target_size, target_size), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(target_size * target_size, embed_dim)
            if model_pos_embed.shape[0] == patch_pos.shape[0] + 1:
                new_pos_embed = torch.cat([cls_token, patch_pos], dim=0)
            else:
                new_pos_embed = patch_pos
            self.positional_embedding.data.copy_(new_pos_embed)
        msg = self.load_state_dict(new_state_dict, strict=False)
        logger = MMLogger.get_current_instance()
        logger.info(msg)
        print(f'[SARCLIPViT] loaded checkpoint from {checkpoint}')
        if hasattr(msg, 'missing_keys') and msg.missing_keys:
            print(f'[SARCLIPViT] missing keys ({len(msg.missing_keys)}): {msg.missing_keys}')
        if hasattr(msg, 'unexpected_keys') and msg.unexpected_keys:
            print(f'[SARCLIPViT] unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys}')

    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        x = self.conv1(x)
        patch_h, patch_w = x.shape[2], x.shape[3]

        x = x.reshape(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        cls_token = self.class_embedding.to(dtype=x.dtype, device=x.device).view(1, 1, -1).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self._interpolate_pos_embed(height, width, x.dtype, x.device)
        x = self.ln_pre(x)

        _, token_features = self.transformer(x, out_indices=set(self.out_indices))
        features = []
        for feature in token_features:
            feature = feature[:, 1:, :].permute(0, 2, 1).reshape(batch_size, self.embed_dim, patch_h, patch_w).contiguous()
            features.append(feature)

        if self.with_fpn:
            pyramid_ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            features = [op(feature) for op, feature in zip(pyramid_ops, features)]

        return tuple(features)
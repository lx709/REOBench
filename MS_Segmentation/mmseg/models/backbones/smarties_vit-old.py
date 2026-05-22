"""
SMARTIES Vision Transformer Backbone for MMSegmentation

This is a minimal adaptation of SMARTIES ViT for semantic segmentation.
The projection indices (proj_indices) for DFC2020 are hardcoded for simplicity.

Original SMARTIES: https://github.com/NASA-IMPACT/SMARTIES
"""
from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS


# ============================================================================
# Utility Functions from SMARTIES
# ============================================================================

def tensor_patchify(imgs, patch_size):
    """
    Patchify images into patches.

    Args:
        imgs: (N, C, H, W)
        patch_size: size of each patch

    Returns:
        patches: (N, H', W', C, p, p) where H'=H//p, W'=W//p, p=patch_size
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h, w, p, p, imgs.shape[1])).permute(0, 1, 2, 5, 3, 4)
    return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sine-cosine position embeddings.

    Args:
        embed_dim: embedding dimension
        grid_size: int of the grid height and width
        cls_token: if True, add cls token position

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Get 2D position embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Get 1D position embedding."""
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


# ============================================================================
# Spectrum-Aware Projection Modules
# ============================================================================

class SpectrumRangeProjection(nn.Module):
    """Project a single spectral range to embedding space."""

    def __init__(self, spectral_range, spectrum_spec, patch_size, embed_dim, bias=True):
        super().__init__()
        self.spectral_range = spectral_range
        self.name = spectrum_spec['name']
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.sensors = spectrum_spec['sensors']
        self.nb_pixels = patch_size**2
        self.proj = nn.Linear(self.nb_pixels, embed_dim, bias=bias)

    def forward(self, x):
        return self.proj(x.view(-1, self.nb_pixels))


class SpectrumRangeProjectionAvg(nn.Module):
    """Aggregate multiple spectral projections with weighted average."""

    def __init__(self, spectrum_projections, spectrum_spec, embed_dim):
        super().__init__()
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.central_lambda = 0.5 * (float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections

        # Calculate weights based on wavelength distance
        weights = []
        for spectrum_proj in self.spectrum_projections:
            central_lambda = 0.5 * (float(spectrum_proj.min_wavelength) +
                                   float(spectrum_proj.max_wavelength))
            weights.append(abs(self.central_lambda - central_lambda))
        self.weights = np.array(weights) / sum(weights)
        self.embed_dim = embed_dim

    def forward(self, x):
        out = 0.
        for i, spectrum_proj in enumerate(self.spectrum_projections):
            out += spectrum_proj(x) * self.weights[i]
        return out


class SpectrumAwareProjection(nn.Module):
    """Spectrum-aware projection for multi-spectral inputs."""

    def __init__(self, spectrum_specs, patch_size, embed_dim, bias=True):
        super().__init__()
        self.nb_pixels = patch_size**2

        self.spectrum_embeds = torch.nn.ModuleList()

        # First pass: create base projections
        for spectral_range in sorted(spectrum_specs,
                                     key=lambda key: spectrum_specs[key]['projection_idx']):
            spec = spectrum_specs[spectral_range]
            if spec['projection_idx'] != -1 and len(spec['agg_projections']) == 0:
                self.spectrum_embeds.append(
                    SpectrumRangeProjection(spectral_range, spec, patch_size, embed_dim)
                )

        # Second pass: create aggregated projections
        for spectral_range in sorted(spectrum_specs,
                                     key=lambda key: spectrum_specs[key]['projection_idx']):
            spec = spectrum_specs[spectral_range]
            if spec['projection_idx'] != -1 and len(spec['agg_projections']) > 0:
                agg_projs = [self.spectrum_embeds[idx] for idx in spec['agg_projections']]
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(agg_projs, spec, embed_dim)
                )

    def forward(self, x, projection_idx):
        """
        Args:
            x: input patches
            projection_idx: index of the projection layer to use
        """
        return self.spectrum_embeds[projection_idx](x)


# ============================================================================
# Vision Transformer Components (from timm)
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# SMARTIES Vision Transformer Backbone
# ============================================================================

@MODELS.register_module()
class SmartiesViT(BaseModule):
    """
    SMARTIES Vision Transformer for Multi-spectral Semantic Segmentation.

    This backbone is specifically configured for DFC2020 dataset with
    hardcoded projection indices for minimal code changes.

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_chans (int): Number of input channels. Default: 14 (for DFC2020)
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0
        qkv_bias (bool): Enable bias for qkv. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        out_indices (Sequence[int]): Output from which stages. Default: (11,)
        frozen_stages (int): Stages to be frozen. Default: -1
        use_dfc2020_proj_indices (bool): Use hardcoded DFC2020 proj_indices. Default: True
        pretrained (str): Path to pretrained weights. Default: None
        init_cfg (dict): Initialization config. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=14,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 out_indices=(11,),
                 frozen_stages=-1,
                 use_dfc2020_proj_indices=True,
                 pretrained=None,
                 init_cfg=None):
        super(SmartiesViT, self).__init__(init_cfg=init_cfg)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # DFC2020 hardcoded projection indices
        # This maps the 12 selected bands to their spectral projection indices
        if use_dfc2020_proj_indices:
            # From SMARTIES config: [0, 1, 4, 6, 7, 8, 10, 9, 11, 12, 13, 14]
            self.register_buffer(
                'proj_indices',
                torch.tensor([0, 1, 4, 6, 7, 8, 10, 9, 11, 12, 13, 14], dtype=torch.long)
            )
            print(f"SmartiesViT: Using hardcoded DFC2020 projection indices: {self.proj_indices.tolist()}")
        else:
            self.proj_indices = None

        # Load spectrum specs for DFC2020
        self.spectrum_specs = self._load_dfc2020_spectrum_specs()

        # Spectrum-aware projection (replaces patch_embed)
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=self.spectrum_specs,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        # Position embedding (sine-cosine)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            img_size // patch_size,
            cls_token=True
        )
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Projection scaler (from SMARTIES)
        self.projection_scaler = 12

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Norm layer
        self.norm = norm_layer(embed_dim)

        # Freeze stages
        self._freeze_stages()

        # Load pretrained weights if specified
        if pretrained is not None:
            self.init_weights(pretrained)

    def _load_dfc2020_spectrum_specs(self):
        """
        Load DFC2020 spectrum specifications.
        Hardcoded to avoid external dependencies.
        """
        # This is a simplified version with only the necessary spectral ranges for DFC2020
        spectrum_specs = {
            'aerosol': {
                'name': 'B01 (aerosol)',
                'min_wavelength': 422,
                'max_wavelength': 463,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 0,
                'agg_projections': []
            },
            'blue_1': {
                'name': 'B02 (blue)',
                'min_wavelength': 427,
                'max_wavelength': 558,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 1,
                'agg_projections': []
            },
            'blue_3': {
                'name': 'blue',
                'min_wavelength': 430,
                'max_wavelength': 545,
                'sensors': ['RGB'],
                'projection_idx': 2,
                'agg_projections': []
            },
            'green_1': {
                'name': 'green',
                'min_wavelength': 466,
                'max_wavelength': 620,
                'sensors': ['RGB'],
                'projection_idx': 3,
                'agg_projections': []
            },
            'green_2': {
                'name': 'B03 (green)',
                'min_wavelength': 524,
                'max_wavelength': 595,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 4,
                'agg_projections': []
            },
            'red_1': {
                'name': 'red',
                'min_wavelength': 590,
                'max_wavelength': 710,
                'sensors': ['RGB'],
                'projection_idx': 5,
                'agg_projections': []
            },
            'red_2': {
                'name': 'B04 (red)',
                'min_wavelength': 634,
                'max_wavelength': 696,
                'sensors': ['SENTINEL2'],
                'projection_idx': 6,
                'agg_projections': []
            },
            'red_edge_1': {
                'name': 'B05 (red edge 1)',
                'min_wavelength': 689,
                'max_wavelength': 719,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 7,
                'agg_projections': []
            },
            'red_edge_2': {
                'name': 'B06 (red edge 2)',
                'min_wavelength': 726,
                'max_wavelength': 755,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 8,
                'agg_projections': []
            },
            'near_infrared_1': {
                'name': 'B08 (NIR 1)',
                'min_wavelength': 728,
                'max_wavelength': 938,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 9,
                'agg_projections': []
            },
            'near_infrared_2': {
                'name': 'B07 (NIR 2)',
                'min_wavelength': 761,
                'max_wavelength': 802,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 10,
                'agg_projections': []
            },
            'near_infrared_3': {
                'name': 'B8A (NIR 3)',
                'min_wavelength': 843,
                'max_wavelength': 886,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 11,
                'agg_projections': []
            },
            'short_wave_infrared_1': {
                'name': 'B09 (SWIR water vapour)',
                'min_wavelength': 923,
                'max_wavelength': 964,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 12,
                'agg_projections': []
            },
            'short_wave_infrared_3': {
                'name': 'B11 (SWIR 1)',
                'min_wavelength': 1516,
                'max_wavelength': 1704,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 13,
                'agg_projections': []
            },
            'short_wave_infrared_4': {
                'name': 'B12 (SWIR 2)',
                'min_wavelength': 2002,
                'max_wavelength': 2376,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'projection_idx': 14,
                'agg_projections': []
            },
            'microwave_1': {
                'name': 'VV',
                'min_wavelength': 5.5e7,
                'max_wavelength': 5.6e7,
                'sensors': ['SENTINEL1-GRD'],
                'projection_idx': 15,
                'agg_projections': []
            },
            'microwave_2': {
                'name': 'VH',
                'min_wavelength': 5.5e7,
                'max_wavelength': 5.6e7,
                'sensors': ['SENTINEL1-GRD'],
                'projection_idx': 16,
                'agg_projections': []
            }
        }
        return spectrum_specs

    def _freeze_stages(self):
        """Freeze stages of the model."""
        if self.frozen_stages >= 0:
            self.spectrum_projection.eval()
            for param in self.spectrum_projection.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize weights from pretrained checkpoint."""
        if pretrained is not None:
            print(f'Loading pretrained model from {pretrained}')
            checkpoint = CheckpointLoader.load_checkpoint(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Load state dict
            msg = self.load_state_dict(state_dict, strict=False)
            print(f'Missing keys: {msg.missing_keys}')
            print(f'Unexpected keys: {msg.unexpected_keys}')

    def forward_features(self, imgs, proj_indices=None):
        """
        Extract features from multi-spectral images.

        Args:
            imgs: (B, C, H, W) - multi-spectral images
            proj_indices: (C,) or (B, C) - projection indices for each band
                         If None, uses hardcoded DFC2020 indices

        Returns:
            list of feature tensors from specified output stages
        """
        # Use hardcoded proj_indices if not provided
        if proj_indices is None:
            if self.proj_indices is None:
                raise ValueError("proj_indices must be provided or use_dfc2020_proj_indices=True")
            proj_indices = self.proj_indices

        # Ensure proj_indices is on the same device
        if not isinstance(proj_indices, torch.Tensor):
            proj_indices = torch.tensor(proj_indices, device=imgs.device, dtype=torch.long)
        else:
            proj_indices = proj_indices.to(imgs.device)

        # Patchify images: (B, C, H, W) -> (B, H', W', C, p, p)
        img_patches = tensor_patchify(imgs, self.patch_size)
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        # Initialize spectrum embeddings
        img_spectrum_embeds = torch.zeros(
            (B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim),
            device=device,
            dtype=imgs.dtype
        )

        # Apply spectrum-aware projections
        for projection_idx in torch.unbind(torch.unique(proj_indices)):
            mask = (proj_indices == projection_idx)
            img_spectrum_embeds[:, :, :, mask] = self.spectrum_projection(
                img_patches[:, :, :, mask], projection_idx
            ).view(B, nb_patch_h, nb_patch_w, -1, self.embed_dim)

        # Average across bands and reshape
        img_embeddings = self.projection_scaler * img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1, nb_patch_h * nb_patch_w, self.embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, img_embeddings), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks and collect features from specified stages
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove CLS token and reshape to spatial format
                # (B, N, C) -> (B, C, H', W')
                out = x[:, 1:, :].permute(0, 2, 1).reshape(
                    -1, self.embed_dim, nb_patch_h, nb_patch_w
                )
                outs.append(out)

        # Apply norm to last output
        if len(outs) > 0:
            outs[-1] = self.norm(outs[-1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return tuple(outs)

    def forward(self, x):
        """
        Forward function for semantic segmentation.

        Args:
            x: (B, C, H, W) - input images

        Returns:
            tuple of feature maps
        """
        # Use hardcoded DFC2020 proj_indices
        return self.forward_features(x, proj_indices=self.proj_indices)

    def train(self, mode=True):
        """Set module to training mode."""
        super(SmartiesViT, self).train(mode)
        self._freeze_stages()


# Convenience functions for different model sizes

@MODELS.register_module()
class SmartiesViT_Base(SmartiesViT):
    """SMARTIES ViT-Base: 12 layers, 768 dim, 12 heads."""

    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            **kwargs
        )


@MODELS.register_module()
class SmartiesViT_Large(SmartiesViT):
    """SMARTIES ViT-Large: 24 layers, 1024 dim, 16 heads."""

    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            **kwargs
        )


@MODELS.register_module()
class SmartiesViT_Huge(SmartiesViT):
    """SMARTIES ViT-Huge: 32 layers, 1280 dim, 16 heads."""

    def __init__(self, **kwargs):
        super().__init__(
            patch_size=14,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            **kwargs
        )

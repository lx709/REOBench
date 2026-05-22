# Copyright (c) OpenMMLab. All rights reserved.
"""SMARTIES Vision Transformer Backbone.

Migrated from SMARTIES project for mmpretrain framework.
Supports spectrum-aware projection for multi-spectral satellite imagery.
"""
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from mmengine.model import BaseModule
# from mmengine.runner.checkpoint import load_checkpoint
from mmengine.runner import CheckpointLoader
from safetensors.torch import load_file

from mmpretrain.registry import MODELS


# ============================================================================
# Position Embedding Utilities
# ============================================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: int, embedding dimension
        grid_size: int, grid height and width
        cls_token: bool, whether to include class token

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
    """Generate 2D positional embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: output dimension for each position
        pos: positions to be encoded, size (M,)

    Returns:
        emb: (M, D)
    """
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


def tensor_patchify(imgs, patch_size):
    """
    Patchify images into patches.

    Args:
        imgs: (N, C, H, W)
        patch_size: int

    Returns:
        x: (N, H', W', C, patch_size, patch_size) where H'=H//patch_size, W'=W//patch_size
    """
    p = patch_size
    # print(f'image.shape{imgs.shape}')
    # import pdb;pdb.set_trace()
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h, w, p, p, imgs.shape[1])).permute(0, 1, 2, 5, 3, 4)
    return x


def get_dtype(mixed_precision):
    """Get torch dtype from mixed precision setting."""
    if mixed_precision == 'no':
        return torch.float32
    elif mixed_precision == 'bf16':
        return torch.bfloat16
    elif mixed_precision == 'fp16':
        return torch.float16
    else:
        return torch.float32


# ============================================================================
# Spectrum-Aware Projection Modules
# ============================================================================
class SpectrumRangeProjection(nn.Module):
    """Patch Embedding projection for a single spectral range."""

    def __init__(
        self,
        spectral_range,
        spectrum_spec,
        patch_size,
        embed_dim,
        bias=True
    ):
        super().__init__()
        self.spectral_range = spectral_range
        self.name = spectrum_spec['name']
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.sensors = spectrum_spec['sensors']
        self.nb_pixels = patch_size**2
        self.proj = nn.Linear(self.nb_pixels, embed_dim, bias=bias)

    def forward(self, x):
        return self.proj(x.reshape(-1, self.nb_pixels))


class SpectrumRangeProjectionAvg(nn.Module):
    """Averaged projection for spectral ranges without dedicated projection."""

    def __init__(
        self,
        spectrum_projections,
        spectrum_spec,
        embed_dim
    ):
        super().__init__()
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.central_lambda = 0.5 * (float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections
        self.weights = []
        for spectrum_proj in self.spectrum_projections:
            central_lambda = 0.5 * (float(spectrum_proj.min_wavelength) + float(spectrum_proj.max_wavelength))
            self.weights.append(abs(self.central_lambda - central_lambda))
        self.weights = np.array(self.weights) / sum(self.weights)
        self.embed_dim = embed_dim

    def forward(self, x):
        out = 0.
        for i, spectrum_proj in enumerate(self.spectrum_projections):
            out += spectrum_proj(x) * self.weights[i]
        return out


class SpectrumAwareProjection(nn.Module):
    """Spectrum-aware projection module for multi-spectral data."""

    def __init__(
        self,
        spectrum_specs,
        patch_size,
        embed_dim,
        bias=True
    ):
        super().__init__()
        self.nb_pixels = patch_size**2

        self.spectrum_embeds = torch.nn.ModuleList()
        for spectral_range in sorted(spectrum_specs, key=lambda key: spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and
                (len(spectrum_specs[spectral_range]['agg_projections']) == 0)):
                self.spectrum_embeds.append(SpectrumRangeProjection(
                    spectral_range, spectrum_specs[spectral_range], patch_size, embed_dim
                ))

        for spectral_range in sorted(spectrum_specs, key=lambda key: spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and
                (len(spectrum_specs[spectral_range]['agg_projections']) > 0)):
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(
                        [self.spectrum_embeds[agg_proj_idx] for agg_proj_idx in spectrum_specs[spectral_range]['agg_projections']],
                        spectrum_specs[spectral_range],
                        embed_dim))

    def forward(self, x, projection_idx):
        return self.spectrum_embeds[projection_idx](x)


# ============================================================================
# SMARTIES Vision Transformer
# ============================================================================
class SmartiesVisionTransformer(TimmVisionTransformer):
    """
    SMARTIES Vision Transformer with spectrum-aware projection.

    This model handles multi-spectral satellite imagery by projecting each
    spectral band through learned projections that are aware of wavelength ranges.
    """

    def __init__(
        self,
        global_pool=False,
        all_tokens=False,
        spectrum_specs=None,
        multi_modal=False,
        num_sources=1,
        mixed_precision='no',
        **kwargs
    ):
        super(SmartiesVisionTransformer, self).__init__(**kwargs)
        del self.patch_embed

        self.dtype = get_dtype(mixed_precision)
        self.patch_size = kwargs['patch_size']
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=spectrum_specs,
            patch_size=self.patch_size,
            embed_dim=kwargs["embed_dim"]
        )

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(kwargs['img_size'] / self.patch_size),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.projection_scaler = 12
        self.all_tokens = all_tokens
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            self.fc_norm = norm_layer(kwargs["embed_dim"])
            del self.norm

        if multi_modal:
            del self.head
            self.num_sources = num_sources
            self.head = nn.Linear(self.embed_dim * self.num_sources, kwargs['num_classes'])
        self.multi_modal = multi_modal

    def forward_encoder(self, batch, is_patchify):
        """
        Forward pass through encoder.

        Args:
            batch: tuple of (imgs, proj_indices)
                imgs: (B, C, H, W) input images
                proj_indices: (B, C) projection indices for each channel
            is_patchify: bool, whether to patchify input

        Returns:
            outcome: encoded features
        """
        imgs, proj_indices = batch
        if is_patchify:
            img_patches = tensor_patchify(imgs, self.patch_size)
        else:
            img_patches = imgs
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim),
                                         device=device, dtype=self.dtype)

        
        # Process each band separately based on its projection index
        # for band_idx in range(nb_bands):
        #     # Get projection index for this band (same for all samples in batch)
        #     projection_idx = proj_indices[0, band_idx].item()
        #     # Project all patches of this band across all samples
        #     band_patches = img_patches[:, :, :, band_idx, :, :]  # (B, nb_patch_h, nb_patch_w, patch_size, patch_size)
        #     band_embeds = self.spectrum_projection(band_patches, projection_idx)  # (B*nb_patch_h*nb_patch_w, embed_dim)
        #     # Reshape and assign
        #     img_spectrum_embeds[:, :, :, band_idx, :] = band_embeds.reshape(B, nb_patch_h, nb_patch_w, self.embed_dim)
                # Build projection_idx to module_idx mapping (once) at parent class level
        if not hasattr(SmartiesViT, '_proj_idx_to_module_cache'):
            sorted_proj_indices = sorted(set(SmartiesViT.BIGEARTHNET_S2_PROJ_INDICES.tolist()))
            SmartiesViT._proj_idx_to_module_cache = {proj_idx: module_idx for module_idx, proj_idx in enumerate(sorted_proj_indices)}

        # Process each band separately based on its projection index
        for band_idx in range(nb_bands):
            # Get projection index for this band
            projection_idx = proj_indices[0, band_idx].item()
            # Map projection_idx to module index in spectrum_embeds
            module_idx = SmartiesViT._proj_idx_to_module_cache[projection_idx]
            # Project all patches of this band across all samples
            band_patches = img_patches[:, :, :, band_idx, :, :]  # (B, nb_patch_h, nb_patch_w, patch_size, patch_size)
            band_embeds = self.spectrum_projection(band_patches, module_idx)  # (B*nb_patch_h*nb_patch_w, embed_dim)
            # Reshape and assign
            img_spectrum_embeds[:, :, :, band_idx, :] = band_embeds.reshape(B, nb_patch_h, nb_patch_w, self.embed_dim)

        img_embeddings = self.projection_scaler * img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1, nb_patch_h * nb_patch_w, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, img_embeddings), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.all_tokens:
            return x[:, 1:, :].permute(0, 2, 1).reshape(-1, self.embed_dim, nb_patch_h, nb_patch_w)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, is_patchify=False):
        """Forward pass."""
        if not self.multi_modal:
            x = self.forward_encoder(x, is_patchify)
        else:
            feats = []
            for i in range(self.num_sources):
                feats.append(self.forward_encoder(x[i], is_patchify))
            if self.all_tokens:
                x = torch.cat(feats, dim=-3)
            else:
                x = torch.cat(feats, dim=-1)
        x = self.head(x)
        return x


# ============================================================================
# mmpretrain Wrapper
# ============================================================================
@MODELS.register_module()
class SmartiesViT(BaseModule):
    """
    SMARTIES Vision Transformer Backbone for mmpretrain.

    This backbone wraps the SmartiesVisionTransformer to work with mmpretrain's
    training and evaluation pipeline. It handles BigEarthNet-S2 data with
    pre-defined projection indices.

    Args:
        arch (str): Architecture size, one of ['base', 'large', 'huge']
        img_size (int): Input image size. Default: 120
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 12 (for BigEarthNet-S2)
        global_pool (bool): Use global pooling. Default: False
        mixed_precision (str): Mixed precision mode. Default: 'no'
        pretrained (str): Path to pretrained weights. Default: None
        init_cfg (dict): Initialization config. Default: None
    """

    # BigEarthNet-S2 spectrum specifications
    # Based on electromagnetic_spectrum.yaml for SENTINEL2 sensor
    BIGEARTHNET_S2_SPECTRUM_SPECS = {
        'aerosol': {
            'min_wavelength': 422, 'max_wavelength': 463,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B01 (aerosol)', 'projection_idx': 0, 'agg_projections': []
        },
        'blue_1': {
            'min_wavelength': 427, 'max_wavelength': 558,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B02 (blue)', 'projection_idx': 1, 'agg_projections': []
        },
        'green_2': {
            'min_wavelength': 524, 'max_wavelength': 595,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B03 (green)', 'projection_idx': 4, 'agg_projections': []
        },
        'red_2': {
            'min_wavelength': 634, 'max_wavelength': 696,
            'sensors': ['SENTINEL2'],
            'name': 'B04 (red)', 'projection_idx': 6, 'agg_projections': []
        },
        'red_edge_1': {
            'min_wavelength': 689, 'max_wavelength': 719,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B05 (red edge 1)', 'projection_idx': 7, 'agg_projections': []
        },
        'red_edge_2': {
            'min_wavelength': 726, 'max_wavelength': 755,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B06 (red edge 2)', 'projection_idx': 8, 'agg_projections': []
        },
        'near_infrared_2': {
            'min_wavelength': 761, 'max_wavelength': 802,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B07 (NIR 2)', 'projection_idx': 10, 'agg_projections': []
        },
        'near_infrared_1': {
            'min_wavelength': 728, 'max_wavelength': 938,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B08 (NIR 1)', 'projection_idx': 9, 'agg_projections': []
        },
        'near_infrared_3': {
            'min_wavelength': 843, 'max_wavelength': 886,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B8A (NIR 3)', 'projection_idx': 11, 'agg_projections': []
        },
        'short_wave_infrared_1': {
            'min_wavelength': 923, 'max_wavelength': 964,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B09 (SWIR water vapour)', 'projection_idx': 12, 'agg_projections': []
        },
        'short_wave_infrared_3': {
            'min_wavelength': 1516, 'max_wavelength': 1704,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B11 (SWIR 1)', 'projection_idx': 13, 'agg_projections': []
        },
        'short_wave_infrared_4': {
            'min_wavelength': 2002, 'max_wavelength': 2376,
            'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
            'name': 'B12 (SWIR 2)', 'projection_idx': 14, 'agg_projections': []
        },
    }

    # BigEarthNet-S2 band order and their projection indices
    # Order: [aerosol, blue_1, green_2, red_2, red_edge_1, red_edge_2,
    #         near_infrared_2, near_infrared_1, near_infrared_3,
    #         short_wave_infrared_1, short_wave_infrared_3, short_wave_infrared_4]
    BIGEARTHNET_S2_PROJ_INDICES = torch.tensor([0, 1, 4, 6, 7, 8, 10, 9, 11, 12, 13, 14], dtype=torch.long)

    arch_settings = {
        'base': {
            'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4
        },
        'large': {
            'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4
        },
        'huge': {
            'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4
        },
    }

    def __init__(
        self,
        arch='base',
        img_size=120,
        patch_size=16,
        in_channels=12,
        global_pool=False,
        mixed_precision='no',
        pretrained=None,
        init_cfg=None,
    ):
        super(SmartiesViT, self).__init__(init_cfg=init_cfg)

        if arch not in self.arch_settings:
            raise ValueError(f'Unsupported arch {arch}, please choose from {list(self.arch_settings.keys())}')

        arch_cfg = self.arch_settings[arch]

        self.model = SmartiesVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0,  # No classification head in backbone
            embed_dim=arch_cfg['embed_dim'],
            depth=arch_cfg['depth'],
            num_heads=arch_cfg['num_heads'],
            mlp_ratio=arch_cfg['mlp_ratio'],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            global_pool=global_pool,
            all_tokens=False,
            spectrum_specs=self.BIGEARTHNET_S2_SPECTRUM_SPECS,
            multi_modal=False,
            num_sources=1,
            mixed_precision=mixed_precision,
        )

        self.in_channels = in_channels
        self.embed_dim = arch_cfg['embed_dim']

        # if pretrained is not None:
        #     # load_checkpoint(self, pretrained, map_location='cpu', strict=False)
        self.pretrained = pretrained
        if self.pretrained is not None:
            self.load_pretrained_weights()
    
    def load_pretrained_weights(self):
        """Load pretrained weights from safetensors file."""
        print(f'Loading pretrained weights from: {self.pretrained}')

        if self.pretrained.endswith('.safetensors'):
            # Load from safetensors
            state_dict = load_file(self.pretrained)
        else:
            # Load from regular checkpoint
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

        # Remove head weights as they're task-specific
        keys_to_remove = []
        for k in state_dict.keys():
            if k.startswith('head.'):
                keys_to_remove.append(k)

        for k in keys_to_remove:
            print(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]

        # Handle pos_embed interpolation if size doesn't match
        if 'pos_embed' in state_dict:
            posemb = state_dict['pos_embed']
            posemb_new = self.model.pos_embed
            if posemb.size() != posemb_new.size():
                print(f'Resizing pos_embed from {posemb.size()} to {posemb_new.size()}')
                # Interpolate position embeddings
                # posemb: [1, old_num_patches+1, embed_dim]
                # posemb_new: [1, new_num_patches+1, embed_dim]

                # Separate class token and position embeddings
                posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:]

                # Get old grid size (assume square grid)
                gs_old = int(np.sqrt(len(posemb_grid[0])))
                gs_new = int(np.sqrt(len(posemb_new[0]) - 1))

                # Reshape to 2D grid and interpolate
                posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                posemb_grid = nn.functional.interpolate(
                    posemb_grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
                posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

                # Concatenate class token and interpolated position embeddings
                posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
                state_dict['pos_embed'] = posemb
                print(f'Resized pos_embed to {posemb.size()}')

        # Load weights
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights: {msg}')

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            tuple: (features,) - output features in tuple format for mmpretrain
        """
        B = x.shape[0]

        # Create projection indices for BigEarthNet-S2
        # Each channel maps to its corresponding projection index
        proj_indices = self.BIGEARTHNET_S2_PROJ_INDICES.unsqueeze(0).expand(B, -1).to(x.device)

        # Forward through SMARTIES encoder
        batch = (x, proj_indices)
        features = self.model.forward_encoder(batch, is_patchify=True)

        # Return as tuple for mmpretrain compatibility
        return (features,)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if self.init_cfg is not None and self.init_cfg.get('type') == 'Pretrained':
            return
        # Position embeddings are already initialized in SmartiesVisionTransformer


# ============================================================================
# Model Builder Functions
# ============================================================================
@MODELS.register_module()
class SmartiesViT_Base(SmartiesViT):
    """SMARTIES ViT-Base backbone."""
    def __init__(self, **kwargs):
        super().__init__(arch='base', **kwargs)


@MODELS.register_module()
class SmartiesViT_Large(SmartiesViT):
    """SMARTIES ViT-Large backbone."""
    def __init__(self, **kwargs):
        super().__init__(arch='large', **kwargs)


@MODELS.register_module()
class SmartiesViT_Huge(SmartiesViT):
    """SMARTIES ViT-Huge backbone."""
    def __init__(self, **kwargs):
        super().__init__(arch='huge', **kwargs)

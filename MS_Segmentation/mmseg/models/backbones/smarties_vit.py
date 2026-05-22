"""
SMARTIES Vision Transformer Backbone for MMSegmentation
Wrapper around the SMARTIES model to make it compatible with mmseg.
"""
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from safetensors.torch import load_file

from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from mmseg.registry import MODELS


# =====================================================================
# Utility functions and classes from SMARTIES model_utils
# =====================================================================

class SpectrumRangeProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
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
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectrum_projections,
            spectrum_spec,
            embed_dim
    ):
        super().__init__()
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.central_lambda = 0.5*(float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections
        self.weights = []
        for spectrum_proj in self.spectrum_projections:
            central_lambda = 0.5*(float(spectrum_proj.min_wavelength) + float(spectrum_proj.max_wavelength))
            self.weights.append(abs(self.central_lambda-central_lambda))
        self.weights = np.array(self.weights) / sum(self.weights)
        self.embed_dim = embed_dim

    def forward(self, x):
        out = 0.
        for i, spectrum_proj in enumerate(self.spectrum_projections):
            out += spectrum_proj(x) * self.weights[i]
        return out


class SpectrumAwareProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
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
        self.proj_idx_to_layer_idx = {}  # Map projection_idx to layer index

        layer_idx = 0
        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) == 0)) :
                self.spectrum_embeds.append(SpectrumRangeProjection(
                    spectral_range, spectrum_specs[spectral_range], patch_size, embed_dim
                ))
                self.proj_idx_to_layer_idx[spectrum_specs[spectral_range]['projection_idx']] = layer_idx
                layer_idx += 1

        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) > 0)):
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(
                        [self.spectrum_embeds[agg_proj_idx] for agg_proj_idx in spectrum_specs[spectral_range]['agg_projections']],
                        spectrum_specs[spectral_range],
                        embed_dim))
                self.proj_idx_to_layer_idx[spectrum_specs[spectral_range]['projection_idx']] = layer_idx
                layer_idx += 1

    def forward(self, x, projection_idx):
        layer_idx = self.proj_idx_to_layer_idx[int(projection_idx)]
        return self.spectrum_embeds[layer_idx](x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def tensor_patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h, w, p, p, imgs.shape[1])).permute(0,1,2,5,3,4)
    return x


def get_dtype(mixed_precision):
    """Get dtype based on mixed precision setting."""
    if mixed_precision == 'no':
        return torch.float32
    elif mixed_precision == 'fp16':
        return torch.float16
    elif mixed_precision == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32


class SmartiesVisionTransformer(nn.Module):
    """SMARTIES Vision Transformer.

    This is adapted from the original SMARTIES model with modifications
    to output multi-scale features for dense prediction tasks.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        global_pool=False,
        all_tokens=True,
        spectrum_specs=None,
        multi_modal=False,
        num_sources=1,
        mixed_precision='no',
        out_indices=[3, 5, 7, 11],
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.all_tokens = all_tokens
        self.global_pool = global_pool
        self.out_indices = out_indices
        self.dtype = get_dtype(mixed_precision)

        # Spectrum-aware projection instead of standard patch embedding
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=spectrum_specs,
            patch_size=self.patch_size,
            embed_dim=embed_dim
        )

        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(img_size / self.patch_size),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        import timm.models.vision_transformer as vit
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            vit.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        self.projection_scaler = 12

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        # Head for classification (not used in segmentation)
        if multi_modal:
            self.num_sources = num_sources
            self.head = nn.Linear(self.embed_dim * self.num_sources, num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        self.multi_modal = multi_modal

    def forward_encoder(self, batch, is_patchify):
        """Forward pass through the encoder."""
        imgs, proj_indices = batch
        if is_patchify:
            img_patches = tensor_patchify(imgs, self.patch_size)
        else:
            img_patches = imgs

        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        img_spectrum_embeds = torch.zeros(
            (B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim),
            device=device,
            dtype=self.dtype
        )

        # Apply spectrum-aware projection
        for projection_idx in torch.unbind(torch.unique(proj_indices)):
            mask = (proj_indices == projection_idx)
            # Apply projection to matching bands
            for band_idx in torch.where(mask)[0]:
                # Extract patches for this band: shape [B, nb_patch_h, nb_patch_w, patch_size, patch_size]
                band_patches = img_patches[:, :, :, band_idx, :, :]
                # Reshape to [B*nb_patch_h*nb_patch_w, patch_size, patch_size]
                band_patches_flat = band_patches.reshape(-1, self.patch_size, self.patch_size)
                # Apply projection: returns [B*nb_patch_h*nb_patch_w, embed_dim]
                proj_result = self.spectrum_projection(band_patches_flat, projection_idx)
                # Reshape back to [B, nb_patch_h, nb_patch_w, embed_dim]
                proj_result = proj_result.reshape(B, nb_patch_h, nb_patch_w, self.embed_dim)
                img_spectrum_embeds[:, :, :, band_idx, :] = proj_result

        # Average over bands
        img_embeddings = self.projection_scaler * img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1, nb_patch_h * nb_patch_w,
                                                self.embed_dim)

        # Add CLS token and positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, img_embeddings), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Collect intermediate features
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove CLS token and reshape to (B, embed_dim, H, W)
                feat = x[:, 1:, :].permute(0, 2, 1).reshape(
                    -1, self.embed_dim, nb_patch_h, nb_patch_w)
                features.append(feat)

        return features

    def forward(self, x, is_patchify=False):
        """Forward pass.

        Args:
            x: tuple of (imgs, proj_indices)
            is_patchify: whether to patchify the input

        Returns:
            List of feature maps at different scales
        """
        if not self.multi_modal:
            features = self.forward_encoder(x, is_patchify)
        else:
            # Multi-modal not implemented for segmentation yet
            raise NotImplementedError(
                "Multi-modal not supported for segmentation")

        return features


@MODELS.register_module()
class SMARTIESViT(BaseModule):
    """SMARTIES Vision Transformer backbone for MMSegmentation.

    Args:
        img_size (int): Input image size. Default: 224.
        patch_size (int): Patch size. Default: 16.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        out_indices (tuple): Indices of output features. Default: (3, 5, 7, 11).
        qkv_bias (bool): Enable bias for qkv. Default: True.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        pretrained (str): Path to pretrained weights. Default: None.
        init_cfg (dict): Config for weight initialization. Default: None.
        spectrum_specs (dict): Spectrum specifications from SMARTIES config.
        mixed_precision (str): Mixed precision mode. Default: 'no'.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=(3, 5, 7, 11),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 spectrum_specs=None,
                 mixed_precision='no',
                 **kwargs):
        super(SMARTIESViT, self).__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.spectrum_specs = spectrum_specs or self._get_default_spectrum_specs()

        self.model = SmartiesVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            global_pool=False,
            all_tokens=True,
            spectrum_specs=self.spectrum_specs,
            multi_modal=False,
            num_sources=1,
            mixed_precision=mixed_precision,
            **kwargs
        )

        # Load pretrained weights if specified
        if self.pretrained is not None:
            self.load_pretrained_weights()

    def _get_default_spectrum_specs(self):
        """Get default spectrum specifications for DFC2020 dataset."""
        # Hardcoded spectrum specs based on DFC2020 configuration
        # Maps band names to their spectral properties
        return {
            'aerosol': {
                'min_wavelength': 422, 'max_wavelength': 463,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B01 (aerosol)', 'projection_idx': 0,
                'agg_projections': []
            },
            'blue_1': {
                'min_wavelength': 427, 'max_wavelength': 558,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B02 (blue)', 'projection_idx': 1,
                'agg_projections': []
            },
            'green_2': {
                'min_wavelength': 524, 'max_wavelength': 595,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B03 (green)', 'projection_idx': 4,
                'agg_projections': []
            },
            'red_2': {
                'min_wavelength': 634, 'max_wavelength': 696,
                'sensors': ['SENTINEL2'],
                'name': 'B04 (red)', 'projection_idx': 6,
                'agg_projections': []
            },
            'red_edge_1': {
                'min_wavelength': 689, 'max_wavelength': 719,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B05 (red edge 1)', 'projection_idx': 7,
                'agg_projections': []
            },
            'red_edge_2': {
                'min_wavelength': 726, 'max_wavelength': 755,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B06 (red edge 2)', 'projection_idx': 8,
                'agg_projections': []
            },
            'near_infrared_2': {
                'min_wavelength': 761, 'max_wavelength': 802,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B07 (NIR 2)', 'projection_idx': 10,
                'agg_projections': []
            },
            'near_infrared_1': {
                'min_wavelength': 728, 'max_wavelength': 938,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B08 (NIR 1)', 'projection_idx': 9,
                'agg_projections': []
            },
            'near_infrared_3': {
                'min_wavelength': 843, 'max_wavelength': 886,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B8A (NIR 3)', 'projection_idx': 11,
                'agg_projections': []
            },
            'short_wave_infrared_1': {
                'min_wavelength': 923, 'max_wavelength': 964,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B09 (SWIR water vapour)', 'projection_idx': 12,
                'agg_projections': []
            },
            'short_wave_infrared_3': {
                'min_wavelength': 1516, 'max_wavelength': 1704,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B11 (SWIR 1)', 'projection_idx': 13,
                'agg_projections': []
            },
            'short_wave_infrared_4': {
                'min_wavelength': 2002, 'max_wavelength': 2376,
                'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
                'name': 'B12 (SWIR 2)', 'projection_idx': 14,
                'agg_projections': []
            },
            'microwave_1': {
                'min_wavelength': 5.5e7, 'max_wavelength': 5.6e7,
                'sensors': ['SENTINEL1-GRD'],
                'name': 'VV', 'projection_idx': 15,
                'agg_projections': []
            },
            'microwave_2': {
                'min_wavelength': 5.5e7, 'max_wavelength': 5.6e7,
                'sensors': ['SENTINEL1-GRD'],
                'name': 'VH', 'projection_idx': 16,
                'agg_projections': []
            },
        }

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
        """Forward function.

        Args:
            x: tuple of (imgs, proj_indices) where:
                - imgs: tensor of shape (B, C, H, W)
                - proj_indices: tensor of shape (num_bands,)

        Returns:
            List of feature maps at different scales.
        """
        # Input x is already in the format (imgs, proj_indices) from the pipeline
        features = self.model(x, is_patchify=True)
        return features


def smarties_vit_base(**kwargs):
    """SMARTIES ViT-Base model."""
    model = SMARTIESViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def smarties_vit_large(**kwargs):
    """SMARTIES ViT-Large model."""
    model = SMARTIESViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def smarties_vit_huge(**kwargs):
    """SMARTIES ViT-Huge model."""
    model = SMARTIESViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
# from utils.model_utils import get_2d_sincos_pos_embed, SpectrumAwareProjection, tensor_patchify
from mmpretrain.registry import MODELS
import numpy as np
from safetensors.torch import load_file

SPECTRUM_SPECS={
    'aerosol': {
        'min_wavelength': 422,
        'max_wavelength': 463,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B01 (aerosol)',
        'projection_idx': 0,
        'agg_projections': []
    },
    'blue_1': {
        'min_wavelength': 427,
        'max_wavelength': 558,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B02 (blue)',
        'projection_idx': 1,
        'agg_projections': []
    },
    'blue_2': {
        'min_wavelength': 452,
        'max_wavelength': 512,
        'sensors': ['Landsat8-L2'],
        'name': 'B2 (blue)',
        'projection_idx': 18,
        'agg_projections': [0, 1]
    },
    'blue_3': {
        'min_wavelength': 430,
        'max_wavelength': 545,
        'sensors': ['RGB'],
        'name': 'blue',
        'projection_idx': 2,
        'agg_projections': []
    },
    'green_1': {
        'min_wavelength': 466,
        'max_wavelength': 620,
        'sensors': ['RGB'],
        'name': 'green',
        'projection_idx': 3,
        'agg_projections': []
    },
    'green_2': {
        'min_wavelength': 524,
        'max_wavelength': 595,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B03 (green)',
        'projection_idx': 4,
        'agg_projections': []
    },
    'red_1': {
        'min_wavelength': 590,
        'max_wavelength': 710,
        'sensors': ['RGB'],
        'name': 'red',
        'projection_idx': 5,
        'agg_projections': []
    },
    'red_2': {
        'min_wavelength': 634,
        'max_wavelength': 696,
        'sensors': ['SENTINEL2'],
        'name': 'B04 (red)',
        'projection_idx': 6,
        'agg_projections': []
    },
    'red_edge_1': {
        'min_wavelength': 689,
        'max_wavelength': 719,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B05 (red edge 1)',
        'projection_idx': 7,
        'agg_projections': []
    },
    'red_edge_2': {
        'min_wavelength': 726,
        'max_wavelength': 755,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B06 (red edge 2)',
        'projection_idx': 8,
        'agg_projections': []
    },
    'near_infrared_1': {
        'min_wavelength': 728,
        'max_wavelength': 938,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B08 (NIR 1)',
        'projection_idx': 9,
        'agg_projections': []
    },
    'near_infrared_2': {
        'min_wavelength': 761,
        'max_wavelength': 802,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B07 (NIR 2)',
        'projection_idx': 10,
        'agg_projections': []
    },
    'near_infrared_3': {
        'min_wavelength': 843,
        'max_wavelength': 886,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B8A (NIR 3)',
        'projection_idx': 11,
        'agg_projections': []
    },
    'short_wave_infrared_1': {
        'min_wavelength': 923,
        'max_wavelength': 964,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B09 (SWIR water vapour)',
        'projection_idx': 12,
        'agg_projections': []
    },
    'short_wave_infrared_2': {
        'min_wavelength': 1345,
        'max_wavelength': 1406,
        'sensors': ['SENTINEL2-L1C'],
        'name': 'B10 (SWIR circus)',
        'projection_idx': -1,
        'agg_projections': []
    },
    'short_wave_infrared_3': {
        'min_wavelength': 1516,
        'max_wavelength': 1704,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B11 (SWIR 1)',
        'projection_idx': 13,
        'agg_projections': []
    },
    'short_wave_infrared_4': {
        'min_wavelength': 2002,
        'max_wavelength': 2376,
        'sensors': ['SENTINEL2-L1C', 'SENTINEL2-L2A'],
        'name': 'B12 (SWIR 2)',
        'projection_idx': 14,
        'agg_projections': []
    },
    'thermal_infrared_1': {
        'min_wavelength': 10600,
        'max_wavelength': 11190,
        'sensors': ['Landsat8-L2'],
        'name': 'B10 (surface temperature)',
        'projection_idx': 17,
        'agg_projections': [14, 15]
    },
    'microwave_1': {
        'min_wavelength': 5.5e7,
        'max_wavelength': 5.6e7,
        'sensors': ['SENTINEL1-GRD'],
        'name': 'VV',
        'projection_idx': 15,
        'agg_projections': []
    },
    'microwave_2': {
        'min_wavelength': 5.5e7,
        'max_wavelength': 5.6e7,
        'sensors': ['SENTINEL1-GRD'],
        'name': 'VH',
        'projection_idx': 16,
        'agg_projections': []
    }
}

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, num_extra_tokens=1):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.pos_embed.shape[-2] - num_extra_tokens

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

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
        return self.proj(x.view(-1, self.nb_pixels)) 

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
        out = 0. #torch.zeros((len(x),self.embed_dim))
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
        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) == 0)) :
                self.spectrum_embeds.append(SpectrumRangeProjection(
                    spectral_range, spectrum_specs[spectral_range], patch_size, embed_dim
                ))

        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']): 
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) > 0)):
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(
                        [self.spectrum_embeds[agg_proj_idx] for agg_proj_idx in spectrum_specs[spectral_range]['agg_projections']], 
                        spectrum_specs[spectral_range],
                        embed_dim))
                
    def forward(self, x, projection_idx):
        return self.spectrum_embeds[projection_idx](x)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
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

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, num_extra_tokens=1):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.pos_embed.shape[-2] - num_extra_tokens

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

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

def apply_label_mixup_fn(batch, mixup_fn, patch_size):
    imgs, img_projection_indices, targets = batch
    imgs, targets = mixup_fn(imgs, targets)
    img_patches = tensor_patchify(imgs, patch_size)
    return (img_patches, img_projection_indices, targets)

def get_dtype(mixed_precision):
    if mixed_precision == 'no':
        return torch.float32
    elif mixed_precision == 'bf16':
        return torch.bfloat16
    elif mixed_precision == 'fp16':
        return torch.float16
    else:
        raise NotImplementedError
    
@MODELS.register_module()
class SmartiesVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self, pretrained=None, global_pool=False, all_tokens=False, spectrum_specs=None, multi_modal=False, num_sources=1, mixed_precision='no', norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    ):
        self.pretrained = pretrained
        super(SmartiesVisionTransformer, self).__init__(**kwargs)
        del self.patch_embed
        self.dtype = get_dtype(mixed_precision)
        self.patch_size = kwargs['patch_size']
        # with open(args.spectrum_specs_path) as f:
        #     spectrum_specs = yaml.safe_load(f.read())
        spectrum_specs = SPECTRUM_SPECS
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=spectrum_specs,
            patch_size=self.patch_size,
            embed_dim=kwargs["embed_dim"]
        )

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(kwargs['img_size']/self.patch_size),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.projection_scaler = 12
        self.all_tokens = all_tokens
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            self.fc_norm = norm_layer(kwargs["embed_dim"])
            del self.norm
        
        if multi_modal:
            del self.head
            self.num_sources = num_sources
            self.head = nn.Linear(self.embed_dim*self.num_sources, kwargs['num_classes'])
        self.multi_modal = multi_modal

        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize weights.

        Args:
            pretrained (str | None): path to checkpoint.
        """

        if self.pretrained is None:
            print('Will train from scratch!')
            return
            # raise RuntimeError(
            #     "DOFA requires pretrained weights, but pretrained=None was given."
            # )

        print(f"loading pretrained weights from {self.pretrained}")

        state_dict = self.state_dict()
        pretrained_state_dict = load_file(self.pretrained)

        for k in ["head.weight", "head.bias"]:
            if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrained_state_dict[k]

        # interpolate position embedding if needed
        try:
            interpolate_pos_embed(self, pretrained_state_dict)
        except Exception as e:
            print("Warning: pos_embed interpolate failed:", e)

        # load weights
        msg = self.load_state_dict(pretrained_state_dict, strict=False)
        print("load state dict msg:")
        print(msg)

    def forward_encoder(self, batch, is_patchify):
        imgs, proj_indices = batch
        if is_patchify:
            img_patches = tensor_patchify(imgs, self.patch_size)
        else:
            img_patches = imgs
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim), device=device, dtype=self.dtype)

        for projection_idx in torch.unbind(torch.unique(proj_indices)):
            mask = (proj_indices==projection_idx)
            img_spectrum_embeds[mask] = self.spectrum_projection(img_patches[mask], projection_idx) 

        img_embeddings = self.projection_scaler*img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1,nb_patch_h*nb_patch_w,self.embed_dim)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, img_embeddings), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.all_tokens:
            return x[:, 1:, :].permute(0,2,1).reshape(-1, self.embed_dim, nb_patch_h, nb_patch_w)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, is_patchify=False):
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
        # x = self.head(x)
        return x

def vit_base_patch16(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

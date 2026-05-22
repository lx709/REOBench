# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
# from .wave_dynamic_layer import (
#     Dynamic_MLP_OFA,
# )

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from mmseg.registry import MODELS

import torch.nn.functional as F
# from src.util.pos_embed import get_1d_sincos_pos_embed_from_grid_torch
import numpy as np


import torch.nn.init as init

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
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
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


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
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
def interpolate_pos_embed(model, checkpoint_model, _num_patches=None):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        if _num_patches is not None:
            num_patches = _num_patches
        else:
            try:
                num_patches = model.patch_embed.num_patches
            except:
                num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
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
                align_corners=True,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


random_seed = 1234
torch.manual_seed(random_seed)


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        torch.manual_seed(42)
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, "Expected 4D input (got {}D input)".format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels, (
            "Expected input to have {} channels (got {} channels)".format(
                self._num_input_channels, channels
            )
        )

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        # y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_Decoder(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=16, decoder_embed=512):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.decoder_embed = decoder_embed
        self._num_kernel = self.kernel_size * self.kernel_size * self.decoder_embed

        # self.weight_generator = nn.Sequential(Basic1d(wv_planes, self.inter_dim, bias=True),
        #                                      nn.Linear(self.inter_dim, self._num_kernel))
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, decoder_embed
        )
        self.scaler = 0.01

        self._init_weights()

    def _get_weights(self, waves, batch=True):
        dweights = []
        dynamic_weights = None
        if batch:
            dynamic_weights = self.weight_generator(waves)
        else:
            for i in range(waves.size(0)):
                dweights.append(self.weight_generator(waves[i]))
            dynamic_weights = torch.stack(dweights, dim=0)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)

    def forward(self, img_feat, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9*16*16,512
        weight, bias = self._get_weights(waves)  # 9,16*16*512
        dynamic_weight = weight.view(
            inplanes * self.kernel_size * self.kernel_size, self.decoder_embed
        )  # 9*16*16,512
        weights = dynamic_weight * self.scaler

        dynamic_out = F.linear(img_feat, weights, bias=None)
        x = dynamic_out
        return x


class Dynamic_Patch_Embed(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.patch_size = (kernel_size, kernel_size)
        self.weight2 = nn.Parameter(
            torch.empty([embed_dim, 2, kernel_size, kernel_size])
        )
        self.bias2 = nn.Parameter(torch.empty([embed_dim]))
        self.weight3 = nn.Parameter(
            torch.empty([embed_dim, 3, kernel_size, kernel_size])
        )
        self.bias3 = nn.Parameter(torch.empty([embed_dim]))
        self.weight4 = nn.Parameter(
            torch.empty([embed_dim, 4, kernel_size, kernel_size])
        )
        self.bias4 = nn.Parameter(torch.empty([embed_dim]))
        self.weight9 = nn.Parameter(
            torch.empty([embed_dim, 9, kernel_size, kernel_size])
        )
        self.bias9 = nn.Parameter(torch.empty([embed_dim]))
        self.weight70 = nn.Parameter(
            torch.empty([embed_dim, 70, kernel_size, kernel_size])
        )
        self.bias70 = nn.Parameter(torch.empty([embed_dim]))
        self.weights = {
            2: self.weight2,
            3: self.weight3,
            4: self.weight4,
            9: self.weight9,
            70: self.weight70,
        }
        self.biass = {
            2: self.bias2,
            3: self.bias3,
            4: self.bias4,
            9: self.bias9,
            70: self.bias70,
        }

    def forward(self, img_feat, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        weights = self.weights[inplanes]
        bias = self.biass[inplanes]

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
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

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)
        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3
        # bias = None

        # dynamic_weight = weight.view(self.embed_dim, inplanes, self.kernel_size, self.kernel_size) #3xoutdx16x16
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


if __name__ == "__main__":
    # num_channels, transformer_dim, patch_depth, patch_height, patch_width
    in_chans = 5
    # pev1 = PatchEmbed()
    # pev2 = PatchEmbed_v2(768, 3, 16, 16)
    # pev3 = BlockwisePatchEmbedding(in_chans, 768, 1, 16, 16)
    inp = torch.randn([5, in_chans, 224, 224])
    inpt = torch.randn([5, 196, 512])
    wave_lengths = torch.tensor(list(range(in_chans))).float()
    wv_planes = 128
    gfl = GaussianFourierFeatureTransform(1, wv_planes // 2, 0.5)
    wvs = wave_lengths.view([in_chans, 1, 1, 1])
    waves1 = gfl(wvs)
    print(waves1.squeeze().shape)
    waves = get_1d_sincos_pos_embed_from_grid_torch(wv_planes, wave_lengths)
    print(waves.shape)
    wg = TransformerWeightGenerator(128, 768 * 256, 768)
    tout = wg(torch.randn([5, 128]))
    # print(tout.view(5,768,16,16).shape)
    # print(pev1(inp).shape)
    # print(pev2(inp).shape)
    # print(pev3(inp).shape)
    # waves, inplanes, wv_planes, kernel_size
    decod = Dynamic_MLP_Decoder(wv_planes, inter_dim=64, kernel_size=16)
    dmlp = Dynamic_MLP_OFA(wv_planes, inter_dim=64, kernel_size=16)
    out = dmlp(inp, wave_lengths)
    dout = decod(inpt, waves)
    uniprompt = torch.randn([1, 1, 128])
    clstoken = torch.randn([1, 1, 128])
    print(dout.shape)

    gfl = GaussianFourierFeatureTransform(1, 5, 0.2)
    gfl2 = GaussianFourierFeatureTransform(1, 5, 0.2)
    x1 = torch.tensor([73.4]).view([1, 1, 1, 1])
    x2 = torch.tensor([74.0]).view([1, 1, 1, 1])
    s1 = gfl(x1).view([1, 10])
    s2 = gfl(x2).view([1, 10])
    s3 = gfl(x1).view([1, 10])
    print(torch.nn.functional.cosine_similarity(s1, s2))
    print(s1)
    print(s3)
    print(s2)


@MODELS.register_module()
class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        pretrained=None,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------
        # prompt setting
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

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
        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self,pretrained=None):
        """Initialize weights.

        Args:
            pretrained (str | None): path to checkpoint.
        """

        if pretrained is None:
            return
            # raise RuntimeError(
            #     "DOFA requires pretrained weights, but pretrained=None was given."
            # )

        print(f"DOFA: loading pretrained weights from {pretrained}")

        checkpoint = torch.load(pretrained, map_location='cpu')

        # interpolate position embedding if needed
        try:
            interpolate_pos_embed(self, checkpoint, self.num_patches)
        except Exception as e:
            print("Warning: pos_embed interpolate failed:", e)

        # load weights
        msg = self.load_state_dict(checkpoint, strict=False)
        print("DOFA load state dict msg:")
        print(msg)

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        # TODO #1 how to convert coordinates to higher dimension
        x, _ = self.patch_embed(x, self.waves)

        hw = self.img_size // self.patch_embed.kernel_size
        hw_shape = (hw, hw)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out_features = []

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(out)

        return out_features

    def forward(self, x, wave_list=[0.06, 0.49, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]):
        # print(">>> backbone.forward() input x type:", type(x))
        # print(">>> backbone.forward() input x len:", len(x) if isinstance(x, (list, tuple)) else "not list")
        # if isinstance(x, list):
        #     print(">>> backbone.forward() input x len:", len(x))
        #     for i, xi in enumerate(x):
        #         try:
        #             print(f"--- x[{i}] type:", type(xi), "shape:", xi.shape)
        #         except:
        #             print(f"--- x[{i}] type:", type(xi), "no shape attribute")
        # else:
        #     print(">>> backbone.forward() input x shape:", x.shape)
        # x = torch.stack(x, dim=0)
        x = self.forward_features(x, wave_list)
        # x = torch.stack(x,dim=0)
        # print("backbone out shape =", [t.shape for t in x])
        # print(">>> backbone forward return type:", type(x))
        # print(">>> backbone.forward() output x type:", type(x))
        # print(">>> backbone.forward() output x:", x)
        return x


def vit_base_patch16(**kwargs):
    model = OFAViT(
        out_indices=[3, 5, 7, 11],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        out_indices=[7, 11, 15, 23],
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = OFAViT(
        out_indices=[7, 15, 23, 31],
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == "__main__":
    check_point = torch.load(
        "checkpoints/mae_ofa_decp_all_vitb16_224_Ball/checkpoint-99.pth"
    )
    vit_model = vit_base_patch16()
    vit_model.load_state_dict(check_point["model"], strict=False)

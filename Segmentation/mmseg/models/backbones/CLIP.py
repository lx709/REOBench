import torch.nn as nn
import open_clip
import torch
from ..builder import BACKBONES
import time
import torch.nn.functional as F
import logging
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import numpy as np
from mmcv.utils import get_logger
# from ..utils import PatchEmbed



def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger

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

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float64, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x, **kwargs):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         # assert H == self.img_size[0] and W == self.img_size[1], \
#         #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x)
#         Hp, Wp = x.shape[2], x.shape[3]
#
#         x = x.flatten(2).transpose(1, 2)
#         return x, (Hp, Wp)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)
@BACKBONES.register_module()
class CLIP():
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,pretrained=None,model_name=None, **kwargs):
        # super(CLIP, self).__init__(**kwargs)
        print(model_name," CLIP ok")
        # model_name = 'ViT-B-32'
        self.model= open_clip.create_model_and_transforms(model_name)
        self.model = self.model[0]
        # print(type(self.model))
        # time.sleep(5)
        # for k,v in self.model.named_parameters():
        #     print(k)


        # pos_embed = get_2d_sincos_pos_embed(self.model.visual.positional_embedding.shape[-1], int(32 ** .5),
        #                                     cls_token=True)
        # self.model.visual.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # self.model.pos_embed=torch.from_numpy(pos_embed).float().unsqueeze(0)
        # self.model.pretrained = pretrained
        # self.model.global_pool = global_pool

        # embed_dims = 768
        # self.model.patch_embed = PatchEmbed(
        #     img_size=512, patch_size=16, in_chans=3, embed_dim=1024)
        # self.patch_embed = PatchEmbed(
        #     in_channels=3,
        #     embed_dims=embed_dims,
        #     conv_type='Conv2d',
        #     kernel_size=16,
        #     stride=16,
        #     padding='corner',
        #     norm_cfg=False if False else None,
        #     init_cfg=None,
        # )
        # self.model.patch_embed = PatchEmbed(img_size=512, patch_size=16, in_chans=3, embed_dim=embed_dims)
        # self.model.fpn1 = nn.Sequential(
        #     nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
        #     Norm2d(embed_dims),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
        # )
        #
        # self.model.fpn2 = nn.Sequential(
        #     nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
        # )
        #
        # self.model.fpn3 = nn.Identity()
        #
        # self.model.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        device = torch.device('cuda:0')
        self.model.to(device)
        # self.patch_embed.to(device)

        print("init ok")

    # def __call__(self, x):
    #     self.forward(x)

    # def backbone(self,x):
    #     self.forward(x)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print("*"*100)

        # pretrained='/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt'
        # pretrained='/opt/data/private/zsy/RS_workspace/RS5M_ViT-H-14.pt'
        # pretrained='/opt/data/private/zsy/RS_workspace/RemoteCLIP-ViT-L-14.pt'
        pretrained='/opt/data/private/zsy/RS_workspace/RemoteCLIP-ViT-B-32.pt'
        # pretrained=pretrained
        checkpoint = torch.load(pretrained)

        msg = self.model.load_state_dict(checkpoint, strict=False)
        # device = torch.device('cuda:0')
        # self.model = self.model.to(device)
        # trunc_normal_(self.head.weight, std=2e-5)
        # interpolate_pos_embed(self, checkpoint_model)

        print("load_CLIP")
        # self.extract_feat = self.model.extract_feat
        # self.backbone = self.model.backbone
        print('full loaded')


    def forward_features(self, x):
        # print(x)
        # print("we here")
        # print(x.shape)
        # time.sleep(5)
        # features = []
        # features.append(self.model.encode_image(x))
        # B,C,H,W = x.shape
        # features = []
        # for i, blk in enumerate(self.model.visual.transformer.resblocks):
        #     x=blk(x)
        #     x=F.normalize(x, dim=-1) if normalize else features
        #     if i in [3,5,7,11]:
        #         features.append(x)

        # B, C, H, W = x.shape
        # x, (Hp, Wp) = self.model.patch_embed(x)
        # print(x.size())
        # batch_size, seq_len, _ = x.size()

        #        cls_tokens = self.cls_token.expand(B, -1, -1)
        #        x = torch.cat((cls_tokens, x), dim=1)
        # if self.model.visual.positional_embedding is not None:
        #     x = x + self.model.visual.positional_embedding[:, 1:, :]
        # x = self.model.pos_drop(x)

        # features = []
        # B,C,H,W = x.shape
        # print("init")
        # print(x.size())
        # x, (Hp,Wp) = self.model.patch_embed(x)
        # print(self.model.pos_embed.size())
        # print(x.size())
        # torch.Size([1, 17, 768])
        # torch.Size([4, 196, 768])

        # if self.model.pos_embed is not None:
        #     x = x + self.model.pos_embed[:, 1:, :]
        # x = self.model.pos_drop(x)

        # stole cls_tokens impl from Phil Wang, thanks
        # cls_tokens = self.model.visual.class_embedding.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = self._pos_embeding(x, hw_shape, self.pos_embed)

        # if not self.with_cls_token:
        # Remove class token for transformer encoder input
        # x = x[:, 1:]
        # for i, blk in enumerate(self.model.visual.transformer.resblocks):
            # print(i)
            # x=blk(x)
            # print(i)
            # print(x.size())
            # x=F.normalize(x, dim=-1)
            # if i in [3,5,7,11]:
                # features.append(x)
        # features.append(self.model.encode_image(x))
        # features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp,Wp), features))
        # ops = [self.model.fpn1, self.model.fpn2, self.model.fpn3, self.model.fpn4]
        # for i in range(len(ops)):
        #     features[i] = ops[i](features[i])

        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if i == len(self.layers) - 1:
        #         if self.final_norm:
        #             x = self.norm1(x)
        #     if i in self.out_indices:
        #         if self.with_cls_token:
        #             # Remove class token and reshape token for decoder head
        #             out = x[:, 1:]
        #         else:
        #             out = x
        #         B, _, C = out.shape
        #         out = out.reshape(B, hw_shape[0], hw_shape[1],
        #                           C).permute(0, 3, 1, 2).contiguous()
        #         if self.output_cls_token:
        #             out = [out, x[:, 0]]
        #         outs.append(out)
        # for i in range(4):
        #     features.append(self.model.visual(x))
        # print(len(features[0][0][0][0][0]))
        return self.model.forward_features(x)


        # features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))
        # for
        # print(features)
        # print("features")
        # print(features.shape)
        # time.sleep(5)
        # print("ffok")
        # return tuple(features)

    def forward(self, x):
        # print("x")
        # print(x)
        x = self.forward_features(x)
        # print("features X")
        # print(x)
        return x

# model=CLIP(model_name="ViT-B-32")
# ones_tensor = torch.ones(4, 3, 224, 224)
# # v1=torch.Size([4, 3, 224, 224])
# v1=torch.tensor(ones_tensor)
# ones_tensor.cuda()
# v1.cuda()
# # model.to(device)
# print(v1.shape)
# t=model(v1)
'''
@BACKBONES.register_module()
class CLIP(open_clip.model.CLIP):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super(open_clip.model.CLIP).__init__()
        # self.output_dict = output_dict
        #
        # self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        #
        # text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        # self.transformer = text.transformer
        # self.context_length = text.context_length
        # self.vocab_size = text.vocab_size
        # self.token_embedding = text.token_embedding
        # self.positional_embedding = text.positional_embedding
        # self.ln_final = text.ln_final
        # self.text_projection = text.text_projection
        # self.text_pool_type = text.pool_type
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        #
        # lshape = [1] if nonscalar_logit_scale else []
        # self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        # if init_logit_bias is not None:
        #     self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        # else:
        #     self.logit_bias = None
        # print(model_name," CLIP ok")
        # model_name = 'ViT-B-32'
        # self.model= open_clip.create_model_and_transforms(model_name)
        # self.model = self.model[0]
        # print(type(self.model))
        # time.sleep(5)
        # for k,v in self.model.named_parameters():
        #     print(k)


        pos_embed = get_2d_sincos_pos_embed(self.model.visual.positional_embedding.shape[-1], int(32 ** .5),
                                            cls_token=True)
        # self.model.visual.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.model.pos_embed=torch.from_numpy(pos_embed).float().unsqueeze(0)
        # self.model.pretrained = pretrained
        # self.model.global_pool = global_pool

        embed_dims = 768
        # self.model.patch_embed = PatchEmbed(
        #     img_size=512, patch_size=16, in_chans=3, embed_dim=1024)
        # self.patch_embed = PatchEmbed(
        #     in_channels=3,
        #     embed_dims=embed_dims,
        #     conv_type='Conv2d',
        #     kernel_size=16,
        #     stride=16,
        #     padding='corner',
        #     norm_cfg=False if False else None,
        #     init_cfg=None,
        # )
        self.patch_embed = PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=embed_dims)
        self.model.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
            Norm2d(embed_dims),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
        )

        self.model.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
        )

        self.model.fpn3 = nn.Identity()

        self.model.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        device = torch.device('cuda')
        self.model.to(device)
        self.patch_embed.to(device)

        print("init ok")

    def __call__(self, x):
        self.forward(x)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print("*"*100)

        pretrained='/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt'
        # pretrained=pretrained
        checkpoint = torch.load(pretrained)

        msg = self.model.load_state_dict(checkpoint, strict=False)
        # device = torch.device('cuda:0')
        # self.model = self.model.to(device)
        # trunc_normal_(self.head.weight, std=2e-5)
        # interpolate_pos_embed(self, checkpoint_model)

        print("load_CLIP")
        # self.extract_feat = self.model.extract_feat
        # self.backbone = self.model.backbone
        print('full loaded')


    def forward_features(self, x):
        # print(x)
        # print("we here")
        # print(x.shape)
        # time.sleep(5)
        # features = []
        # features.append(self.model.encode_image(x))
        # B,C,H,W = x.shape
        # features = []
        # for i, blk in enumerate(self.model.visual.transformer.resblocks):
        #     x=blk(x)
        #     x=F.normalize(x, dim=-1) if normalize else features
        #     if i in [3,5,7,11]:
        #         features.append(x)

        # B, C, H, W = x.shape
        # x, (Hp, Wp) = self.model.patch_embed(x)
        # print(x.size())
        # batch_size, seq_len, _ = x.size()

        #        cls_tokens = self.cls_token.expand(B, -1, -1)
        #        x = torch.cat((cls_tokens, x), dim=1)
        # if self.model.visual.positional_embedding is not None:
        #     x = x + self.model.visual.positional_embedding[:, 1:, :]
        # x = self.model.pos_drop(x)

        features = []
        B,C,H,W = x.shape
        # print("init")
        # print(x.size())
        x, (Hp,Wp) = self.patch_embed(x)
        # print(self.model.pos_embed.size())
        # print(x.size())
        # torch.Size([1, 17, 768])
        # torch.Size([4, 196, 768])

        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed[:, 1:, :]
        # x = self.model.pos_drop(x)

        # stole cls_tokens impl from Phil Wang, thanks
        # cls_tokens = self.model.visual.class_embedding.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = self._pos_embeding(x, hw_shape, self.pos_embed)

        # if not self.with_cls_token:
        # Remove class token for transformer encoder input
        # x = x[:, 1:]
        for i, blk in enumerate(self.model.visual.transformer.resblocks):
            # print(i)
            x=blk(x)
            # print(i)
            # print(x.size())
            # x=F.normalize(x, dim=-1)
            if i in [3,5,7,11]:
                features.append(x)
        # features.append(self.model.encode_image(x))
        features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp,Wp), features))
        ops = [self.model.fpn1, self.model.fpn2, self.model.fpn3, self.model.fpn4]
        for i in range(len(ops)):
            features[i] = ops[i](features[i])

        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if i == len(self.layers) - 1:
        #         if self.final_norm:
        #             x = self.norm1(x)
        #     if i in self.out_indices:
        #         if self.with_cls_token:
        #             # Remove class token and reshape token for decoder head
        #             out = x[:, 1:]
        #         else:
        #             out = x
        #         B, _, C = out.shape
        #         out = out.reshape(B, hw_shape[0], hw_shape[1],
        #                           C).permute(0, 3, 1, 2).contiguous()
        #         if self.output_cls_token:
        #             out = [out, x[:, 0]]
        #         outs.append(out)
        # for i in range(4):
        #     features.append(self.model.visual(x))
        # print(len(features[0][0][0][0][0]))
        return tuple(features)


        # features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))
        # for
        # print(features)
        # print("features")
        # print(features.shape)
        # time.sleep(5)
        # print("ffok")
        # return tuple(features)

    def forward(self, x):
        # print("x")
        # print(x)
        x = self.forward_features(x)
        # print("features X")
        # print(x)
        return x
'''

import torch.nn as nn
import sys
sys.path.append('mmrotate/models/backbones')
import open_clip
import torch
from ..builder import ROTATED_BACKBONES
import time
import torch.nn.functional as F
import logging
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import numpy as np
from functools import partial
import pdb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        print(f'embedding_size: {embedding_size}')
        try:
            num_patches = model.patch_embed.num_patches
            print(f'num_patches: {num_patches}')
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(orig_size, new_size)
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
    elif 'visual.positional_embedding' in checkpoint_model:
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_checkpoint = checkpoint_model['visual.positional_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # print(f'embedding_size: {embedding_size}')
        # print(f'model.visual:\n{model.visual}')

        image_size=model.visual.image_size[0]
        patch_size=model.visual.patch_size[0]
        num_patches = (image_size // patch_size) ** 2
        num_extra_tokens=1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            # pdb.set_trace()
            # print(f'pos_embed_checkpoint.shape: {pos_embed_checkpoint.shape}') #pos_embed_checkpoint.shape: torch.Size([50, 768])
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
            # print(f'extra_tokens.shape: {extra_tokens.shape}') #extra_tokens.shape: torch.Size([1, 768])
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
            # print(f'pos_tokens.shape: {pos_tokens.shape}') pos_tokens.shape: torch.Size([49, 768])
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens.squeeze(0)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            checkpoint_model['visual.positional_embedding'] = new_pos_embed
    else:
        print(f'error interplot pos embed')
        print(model.visual)


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor=14/16, mode='bilinear', align_corners=False):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


@ROTATED_BACKBONES.register_module()
class CLIP(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,pretrained=None,model_name=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        clip_model = open_clip.create_model_and_transforms(model_name)
        self.clip = clip_model[0]

        embed_dim = self.clip.visual.width
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)

        if '14' in model_name:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                InterpolateLayer(scale_factor=3.5/4)
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif '32' in model_name:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

            self.fpn4 = nn.Identity()
        else:
            print(f'please check model name:{model_name}')




    # def __call__(self, x):
    #     self.forward(x)

    # def backbone(self,x):
    #     self.forward(x)

    def init_weights(self, pretrained=None):
        
        pretrained = self.pretrained
        if pretrained is None:
            print("Training from scratch: no pretrained weights loaded.")
            return
        else:
            print(f'Loading pretrained weights from {pretrained}')

        checkpoint = torch.load(pretrained)
        # for k in ['positional_embedding']:
        #     if k in checkpoint and k in self.clip.state_dict():
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint[k]
        #         try:
        #             del self.clip.positional_embedding
        #         except AttributeError:
        #             pass 
        # for name in checkpoint.keys():
        #     print(name)
        # print("checkpoint:\n")
        # for name, param in checkpoint.items():
        #     print(f'Name: {name}, Shape: {param.shape}')

        # model_keys = set(self.clip.state_dict().keys())
        # checkpoint_keys = set(checkpoint.keys())

        # missing_keys = model_keys - checkpoint_keys
        # unexpected_keys = checkpoint_keys - model_keys
        #
        # print(f"missing keys: {missing_keys}" if missing_keys else "no missing keys")
        # print(f"unexpected keys: {unexpected_keys}" if unexpected_keys else "no unexpected keys")
        # pdb.set_trace()
        interpolate_pos_embed(self.clip, checkpoint)

        msg = self.clip.load_state_dict(checkpoint, strict=True)
        print(f'load state dict message:\n{msg}')



    def forward_features(self, x):
        x = self.clip.encode_image(x)

        # N, B, D = x.shape
        # Np = int((N - 1) ** 0.5)
        # xp = x[1:, :, :].permute(1, 2, 0).reshape(B, D, Np, Np)

        if isinstance(x, torch.Tensor):
            N, B, D = x.shape
            Np = int((N - 1) ** 0.5)
            x = x[1:, :, :]
            x = self.norm(x)
            xp = x.permute(1, 2, 0).reshape(B, D, Np, Np)
            # pdb.set_trace()
        elif isinstance(x, list):
            xp = []
            for tensor in x:
                N, B, D = tensor.shape
                Np = int((N - 1) ** 0.5)
                tensor = self.norm(tensor)
                tensor = tensor[1:, :, :].permute(1, 2, 0).reshape(B, D, Np, Np)
                xp.append(tensor)
        else:
            raise ValueError(f"Expected input type torch.Tensor or list, but got {type(x)}")

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        features = []
        # for i in range(len(ops)):
        #     features.append(ops[i](xp))
        if isinstance(xp, torch.Tensor):
            for i in range(len(ops)):
                features.append(ops[i](xp))
        elif isinstance(xp, list):
            for i in range(len(ops)):
                features.append(ops[i](xp[i]))
        else:
            raise ValueError(f"Expected xp to be a torch.Tensor or list, but got {type(xp)}")
        
        torch.cuda.empty_cache()
        return tuple(features)
        # return tuple(xp)

    def forward(self, x):
        x = self.forward_features(x)
        return x

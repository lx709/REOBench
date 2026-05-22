# ============================
# DINOv2 ViT-Base Configuration for mmsegmentation
# ============================

_base_ = [
    './datasets/dfc2020.py',
    './runtimes/default_runtime.py',
    './schedules/schedule_frozenbackbone_epoch.py'
]

# ============================
# Model Configuration
# ============================

# Normalization config
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=None,  # Normalization done in pipeline
        std=None,   # Normalization done in pipeline
        bgr_to_rgb=False,  # Multispectral data, not RGB
        pad_val=0,
        seg_pad_val=255,
        size=(98, 98),
        test_cfg=dict(
        # size=(98, 98),  # val/test pad
        size_divisor=14  # 或者设置为 patch_size 的倍数
    )
        ),
    backbone=dict(
        type='DinoV2ViT',
        img_size=224,
        patch_size=14,
        in_chans=12,  # For Sentinel-2 multispectral bands
        embed_dim=768,  # Base model
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.1,
        out_indices=(4, 6, 10, 11),
        pretrained='pretrained/B13_vitb14_softcon.pth',
    ),
    neck = dict(type='Feature2Pyramid',
            embed_dim=768,
            rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],  # Base model embed_dim
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,  # Base model embed_dim
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

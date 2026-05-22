_base_ = [
    './datasets/dfc2020_smarties.py',
    './runtimes/default_runtime.py',
    './schedules/schedule_frozenbackbone_epoch.py'
]

# Model settings
norm_cfg = dict(type='BN', requires_grad=True)

# DFC2020 specific data preprocessor (no normalization for multispectral)
data_preprocessor = dict(
    type='MultispectralSegDataPreProcessor',
    mean=None,  # Normalization done in pipeline
    std=None,   # Normalization done in pipeline
    bgr_to_rgb=False,  # Multispectral data, not RGB
    pad_val=0,
    seg_pad_val=255,
    size=(96, 96))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SMARTIESViT',
        img_size=96,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained='pretrained/smarties-v1-vitb-dfc2020-linprobe.safetensors',
        # spectrum_specs will use default DFC2020 configuration
    ),
    neck = dict(type='Feature2Pyramid',
            embed_dim=768,
            rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


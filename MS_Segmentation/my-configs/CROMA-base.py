
_base_ = [
    './datasets/dfc2020.py',
    './runtimes/default_runtime.py',
    './schedules/schedule_frozenbackbone_epoch.py'
]
# # Data preprocessor
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,  # Normalization done in pipeline
    std=None,   # Normalization done in pipeline
    bgr_to_rgb=False,  # Multispectral data, not RGB
    pad_val=0,
    seg_pad_val=255,
    size=(96, 96))
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PretrainedCROMA',
        pretrained_path='pretrained/CROMA_base.pt',
        # pretrained=None,
        size='base',
        modality='optical',
        image_resolution=96
    ),
    neck = dict(type='Feature2Pyramid',
            embed_dim=768,
            rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768], # Base
        # in_channels=[1024,1024,1024,1024],# Large
        # in_channels=[1280,1280,1280,1280],# Huge
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
        in_channels=768, #Base
        # in_channels=1280, #Huge
        # in_channels=1024, #Large
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

    test_cfg=dict(
            mode='whole'  
        )
    )



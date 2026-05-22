# ============================
# DINOv2 ViT-Base Configuration for mmpretrain
# ============================

_base_ = [
    # './datasets/BigEarthNet-S2.py',
    # './schedules/ft.py',
    './schedules/schedule.py',
    './runtimes/default_runtime.py'
]

# ============================================================
# Model Configuration
# ============================================================
model = dict(
    type='ImageClassifier',
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
        final_norm=True,
        pretrained='pretrained/B13_vitb14_softcon.pth',
    ),
    # neck=dict(type='LinearNeck', in_channels=768, out_channels=768),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
        # loss=dict(
        #     type='CrossEntropyLoss', 
        #     use_sigmoid=True,  # BCE loss for multi-label
        #     loss_weight=1.0
        # ),
        # loss=dict(
        #     type='AsymmetricLoss',   # 换成 AsymmetricLoss
        #     gamma_pos=0.0,           # 正样本 focusing 参数
        #     gamma_neg=4.0,           # 负样本 focusing 参数
        #     clip=0.05,               # optional probability margin
        #     loss_weight=1.0,
        #     use_sigmoid=True          # 保持多标签 BCE 风格
        # ),
        loss=dict(type='MultiLabelSoftMarginLoss', loss_weight=1.0),
    )
)

# Dataset settings
dataset_type = 'MultiLabelDataset'
data_root = '/datasets/BigEarthNet-S2-v1/BigEarthNet-S2-v1-mmpretrain-version'

# Data pipeline
train_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='stat', use_8_bit=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate', angle=90, prob=1.0, pad_val=0, interpolation='bilinear'),
    dict(type='MultiSpectralResize', scale=224, interpolation='bilinear'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='stat', use_8_bit=True),
    dict(type='MultiSpectralResize', scale=224, interpolation='bilinear'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.json',
        data_prefix='',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/validation.json',
        data_prefix='',
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MicroAveragePrecision'),
    # dict(type='MultiLabelMetric', average='macro'),
    dict(type='MultiLabelMetric', average='micro'),
]
test_evaluator = val_evaluator

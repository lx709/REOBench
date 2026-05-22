_base_ = [
    # './datasets/BigEarthNet-S2.py',
    './schedules/schedule.py',
    # './schedules/ft.py',
    './runtimes/default_runtime.py'
]


# Dataset settings
dataset_type = 'MultiLabelDataset'
data_root = '/datasets/BigEarthNet-S2-v1/BigEarthNet-S2-v1-mmpretrain-version'

# Data pipeline
train_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='percentile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate', angle=90, prob=1.0, pad_val=0, interpolation='bilinear'),
    dict(type='MultiSpectralResize', scale=224, interpolation='bilinear'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='percentile'),
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


# ============================================================
# SMARTIES Vision Transformer Configuration for BigEarthNet-S2
# ============================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SmartiesViT',
        arch='base',  # Options: 'base', 'large', 'huge'
        img_size=224,
        patch_size=16,
        in_channels=12,  # BigEarthNet-S2 has 12 bands
        global_pool=False,
        mixed_precision='no',  # Options: 'no', 'fp16', 'bf16'
        pretrained='pretrained/smarties-v1-vitb-bigearthnets2-finetune.safetensors',  # Path to pretrained weights if available
    ),
    # neck=dict(type='LinearNeck', in_channels=768, out_channels=768),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,  # BigEarthNet has 19 classes
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

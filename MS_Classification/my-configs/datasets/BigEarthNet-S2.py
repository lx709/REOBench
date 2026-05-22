# Dataset settings
dataset_type = 'MultiLabelDataset'
data_root = '/datasets/BigEarthNet-S2-v1/BigEarthNet-S2-v1-mmpretrain-version'

# Data pipeline
train_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='stat'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate', angle=90, prob=1.0, pad_val=0, interpolation='bilinear'),
    # dict(type='MultiSpectralResize', scale=224, interpolation='bilinear'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, channel_first=False),
    dict(type='MultiSpectralNormalize', bands='s2', method='stat'),
    # dict(type='MultiSpectralResize', scale=224, interpolation='bilinear'),
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

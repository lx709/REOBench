# dataset settings for DFC2020
dataset_type = 'DFC2020Dataset'
data_root = '/datasets'

# DFC2020 has 14 bands total, but we only use 12 S2 bands (exclude S1 VV/VH)
# Normalization is done in pipeline using SMARTIES original statistics
crop_size = (96, 96)

# SMARTIES statistics (from eval_datasets.yaml, in [0,1] range)
dfc2020_mean = [0.4908, 0.4883, 0.4875, 0.4870, 0.4898, 0.4956,
                0.4944, 0.4944, 0.4954, 0.4961, 0.4877, 0.4925]
dfc2020_std = [0.2237, 0.2103, 0.2098, 0.2136, 0.2244, 0.2441,
               0.2413, 0.2413, 0.2437, 0.2456, 0.1745, 0.2354]

train_pipeline = [
    # 1. Load multispectral image (14 bands: S2 + S1, uint8 [0, 255])
    dict(type='LoadMultispectralImageFromFile', to_float32=True, imdecode_backend='tifffile'),

    # 2. Select only Sentinel-2 bands (first 12 bands, exclude S1 VV/VH)
    dict(type='SelectBands', band_indices=list(range(12))),

    # 3. First normalize: uint8 [0, 255] -> float [0, 1]
    dict(type='MultispectralNormalize', mean=[0.0]*12, std=[255.0]*12),

    # 4. Second normalize: z-score with SMARTIES statistics
    dict(type='MultispectralNormalize', mean=dfc2020_mean, std=dfc2020_std),

    # 5. Load annotations
    dict(type='LoadMultispectralAnnotations', imdecode_backend='tifffile'),

    # 6. Data augmentation
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # Optional: Add more augmentations
    # dict(type='RandomRotate', prob=0.5, degree=90),

    # 7. Pack data
    dict(type='PackSegInputs')
]

test_pipeline = [
    # 1. Load multispectral image (14 bands: S2 + S1, uint8 [0, 255])
    dict(type='LoadMultispectralImageFromFile', to_float32=True, imdecode_backend='tifffile'),

    # 2. Select only Sentinel-2 bands (first 12 bands, exclude S1 VV/VH)
    dict(type='SelectBands', band_indices=list(range(12))),

    # 3. First normalize: uint8 [0, 255] -> float [0, 1]
    dict(type='MultispectralNormalize', mean=[0.0]*12, std=[255.0]*12),

    # 4. Second normalize: z-score with SMARTIES statistics
    dict(type='MultispectralNormalize', mean=dfc2020_mean, std=dfc2020_std),

    # 5. Load annotations
    dict(type='LoadMultispectralAnnotations', imdecode_backend='tifffile'),

    # 6. Pack data
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/labels'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/labels'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

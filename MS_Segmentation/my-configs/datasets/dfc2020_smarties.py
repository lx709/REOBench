# dataset settings for DFC2020 with SMARTIES model
dataset_type = 'DFC2020Dataset'
data_root = '/datasets'

# DFC2020 has 14 bands total, but we only use 12 S2 bands (exclude S1 VV/VH)
# SMARTIES requires projection indices for each band
crop_size = (96, 96)

# Projection indices for DFC2020 S2 bands based on SMARTIES electromagnetic_spectrum
# B1-aerosol: 0, B2-blue: 1, B3-green: 4, B4-red: 6,
# B5-rededge1: 7, B6-rededge2: 8, B7-rededge3: 9, B8-NIR: 10,
# B8A-narrow_NIR: 11, B9-water_vapor: 12, B11-SWIR1: 13, B12-SWIR2: 14
proj_indices_dfc2020 = [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
dfc2020_mean = [0.4908, 0.4883, 0.4875, 0.4870, 0.4898, 0.4956,
                0.4944, 0.4944, 0.4954, 0.4961, 0.4877, 0.4925]
dfc2020_std = [0.2237, 0.2103, 0.2098, 0.2136, 0.2244, 0.2441,
               0.2413, 0.2413, 0.2437, 0.2456, 0.1745, 0.2354]

train_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, imdecode_backend='tifffile'),
    dict(type='SelectBands', band_indices=list(range(12))),  # Select only first 12 bands (S2)
    dict(type='MultispectralNormalize', mean=[0.0]*12, std=[255.0]*12),
    dict(type='MultispectralNormalize', mean=dfc2020_mean, std=dfc2020_std),
    dict(type='LoadMultispectralAnnotations', imdecode_backend='tifffile'),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.8),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='AddProjIndices', proj_indices=proj_indices_dfc2020),
    dict(type='PackMultispectralSegInputs')
]
test_pipeline = [
    dict(type='LoadMultispectralImageFromFile', to_float32=True, imdecode_backend='tifffile'),
    dict(type='SelectBands', band_indices=list(range(12))),  # Select only first 12 bands (S2)
    dict(type='MultispectralNormalize', mean=[0.0]*12, std=[255.0]*12),
    dict(type='MultispectralNormalize', mean=dfc2020_mean, std=dfc2020_std),
    dict(type='LoadMultispectralAnnotations', imdecode_backend='tifffile'),
    dict(type='AddProjIndices', proj_indices=proj_indices_dfc2020),
    dict(type='PackMultispectralSegInputs')
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

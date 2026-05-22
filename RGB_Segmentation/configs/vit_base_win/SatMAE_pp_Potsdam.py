_base_ = [
    '../_base_/models/SatMAE_vit_large.py',  # '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'PotsdamDataset'
# data_root = '/opt/data/private/zsy/RS_workspace/data/potsdam/'
# data_root = '/opt/data/private/zsy/RS_workspace/corrupted_dataset/Potsdam_corrupted/'
data_root = ''
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871],std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697], to_rgb=True
)
crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=False),
#     dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
train_pipeline = [

    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# t.append(transforms.ToTensor())
#             t.append(transforms.Normalize(mean, std))
#             t.append(
#                 transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
#             )
#             t.append(transforms.RandomHorizontalFlip())
#             return transforms.Compose(t)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=512),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/img_dir/train',
        ann_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/img_dir/val',
        ann_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/ann_dir/val',
        pipeline=val_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='brightness_contrast/severity_1/train/',
    #     ann_dir='label/val/',
    #     pipeline=test_pipeline))
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/img_dir/val',
        ann_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/ann_dir/val',
        pipeline=test_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(
                     num_layers=12,
                     layer_decay_rate=0.9,
                 )
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

optimizer_config = dict(grad_clip=None)

model = dict(
    pretrained='/opt/data/private/zsy/RS_workspace/checkpoint_ViT-L_pretrain_fmow_rgb.pth',
    backbone=dict(
        type='SatMAEVisionTransformer',
        patch_size=16, img_size=512, in_chans=3,
        num_classes=5, drop_path_rate=0.2, global_pool=False,
    ),
    decode_head=dict(
        num_classes=5,
        ignore_index=5
    ),
    auxiliary_head=dict(
        num_classes=5,
        ignore_index=5
    ))

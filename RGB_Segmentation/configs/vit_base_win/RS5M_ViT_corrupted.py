_base_ = [
    '../_base_/models/ViT_Base_CLIP.py',  # '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'PotsdamDataset'
# data_root = '/opt/data/private/zsy/RS_workspace/data/potsdam/'
# data_root = '/opt/data/private/zsy/RS_workspace/corrupted_dataset/Potsdam_corrupted/'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (224,224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(224,224), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
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
    workers_per_gpu=1,
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
        # img_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/img_dir/val',
        # ann_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/ann_dir/val',
        img_dir='/opt/data/private/zsy/RS_workspace/corrupted_dataset/Potsdam_corrupted/translate_image/severity_1/val',
        ann_dir='/opt/data/private/zsy/RS_workspace/corrupted_dataset/Potsdam_corrupted/translate_image/severity_1/label',
        # ann_dir='/opt/data/private/zsy/RS_workspace/data/potsdam/ann_dir/val',
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
    pretrained='/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt',
    # pretrained='/opt/data/private/zsy/RS_workspace/RS5M_ViT-H-14.pt',
    # pretrained='/opt/data/private/zsy/RS_workspace/RemoteCLIP-ViT-L-14.pt',
    backbone=dict(
        type='CLIP',
        model_name='ViT-B-32',
        # model_name='ViT-H-14',
        # model_name='ViT-L-14',
# img_size=512,
#         patch_size=16,
#         drop_path_rate=0.1,
#         out_indices=[3, 5, 7, 11],
#         embed_dim=512,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         use_abs_pos_emb=True
    ),
    decode_head=dict(
        num_classes=5,
        ignore_index=5
    ),
    auxiliary_head=dict(
        num_classes=5,
        ignore_index=5
    ))

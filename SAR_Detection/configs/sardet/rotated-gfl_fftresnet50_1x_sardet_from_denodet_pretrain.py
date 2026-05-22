# RotatedGFL finetuning config for SARDet DOTA-format data (DenoDetV2 pretrained).
# This keeps the FFTResNet backbone and SARDet dataloaders from the existing config,
# but switches to a one-stage GFL detector to align with GFL-like pretraining.

_base_ = ['./oriented-rcnn-le90_r50_fpn_1x_sardet_deno2_freeze_backbone_full.py']

angle_version = 'le90'
num_classes = 6
img_scale = (512, 512)
partition_stride = (8, 8)

model = dict(
    _delete_=True,
    type='mmdet.RetinaNet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[36.50463548417378, 36.50467669785022, 36.50465618752095],
        std=[52.12157845801164, 52.12163789765554, 52.12160835551052],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='FFTResNet',
        img_scale=img_scale,
        partition_stride=partition_stride,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/DenoDetV2/best_coco_bbox_mAP_epoch_12.pth',
            prefix='backbone')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RotatedATSSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='RotatedATSSAssigner',
            topk=9,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(type='mmdet.PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# Backbone weights loaded via init_cfg above; no full model load_from needed.
load_from = None

# Conservative learning rate for detector-family switch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05))

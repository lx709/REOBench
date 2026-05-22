# 迁移自 mmdetection/local_configs/SARDet/gfl_r50_denodet_sardet.py
# 
# 模型信息：
# - 原始Backbone: ResNet50
# - 原始Neck: FrequencySpatialFPN (频域-空间特征金字塔)
# - 原始Detector: GFL (Generalized Focal Loss)
# - 原始框架: mmdetection
# - 迁移框架: mmrotate (针对旋转目标检测调整)
#
# 改动说明：
# [原有设置已注释] 保留原有的mmdetection配置以供参考
# [新增设置] 添加mmrotate框架兼容的配置

# ========== 原始mmdetection配置 (已注释) ==========
# _base_ = [
#     '../_base_/datasets/SARDet_100k.py',
#     '../_base_/schedules/schedule_1x.py', 
#     '../_base_/default_runtime.py'
# ]
# 
# num_classes = 6
# model = dict(
#     type='GFL',
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_size_divisor=32),
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     neck=dict(
#         type='FrequencySpatialFPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_input',
#         num_outs=5,
#         norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
#     bbox_head=dict(
#         type='GFLHead',
#         num_classes=num_classes,
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             ratios=[1.0],
#             octave_base_scale=8,
#             scales_per_octave=1,
#             strides=[8, 16, 32, 64, 128]),
#         loss_cls=dict(
#             type='QualityFocalLoss',
#             use_sigmoid=True,
#             beta=2.0,
#             loss_weight=1.0),
#         loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
#         reg_max=16,
#         loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
#     train_cfg=dict(
#         assigner=dict(type='ATSSAssigner', topk=9),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.6),
#         max_per_img=100))

# ========== 原始数据加载配置 (已注释) ==========
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# 
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PackDetInputs', meta_keys=(...))
# ]

# ========== mmrotate新配置 ==========
_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

num_classes = 6

# ========== 新detector配置 ==========
model = dict(
    # [改动] 使用RotatedGFL detector适配mmrotate框架
    type='RotatedGFL',  # [迁移适配] 从GFL改为RotatedGFL
    data_preprocessor=dict(
        type='RotatedDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    # [保留] ResNet50 backbone配置
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # [迁移] FrequencySpatialFPN neck配置 (已复制到mmrotate)
    neck=dict(
        type='FrequencySpatialFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    # [迁移] GFLHead配置 - 已复制到mmrotate
    bbox_head=dict(
        type='GFLHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # [注释] 原始anchor配置
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     ratios=[1.0],
        #     octave_base_scale=8,
        #     scales_per_octave=1,
        #     strides=[8, 16, 32, 64, 128]),
        # [新增] mmrotate旋转anchor配置
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            angles=None),  # [新增] 旋转角度配置
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6, rotation_invariant=True),  # [新增] 旋转不变NMS
        max_per_img=100))

# ========== 数据集配置 [待调整] ==========
# [原始] 使用SARDet_100K数据集
# [新增] 需要调整为mmrotate格式的旋转目标检测数据集
# [原始] 图像大小: 1024x1024, keep_ratio=False
# [新增] mmrotate推荐配置

# [原始数据加载pipeline]
# train_pipeline = [...]
# test_pipeline = [...]

# [新增mmrotate数据加载pipeline]
# data_root = 'data/split_ss_dota/'  # 根据实际路径修改

print("配置文件加载: mmrotate ResNet50 + FrequencySpatialFPN GFL for SARDet")
print("说明: 此配置基于mmdetection SARDet模型迁移而来")
print("待操作: 1. 调整数据集路径和格式")
print("        2. 验证FrequencySpatialFPN在mmrotate中的兼容性")
print("        3. 根据实际旋转角度范围调整anchor_generator.angles配置")
print("        4. 验证频域-空间特征在旋转检测中的有效性")

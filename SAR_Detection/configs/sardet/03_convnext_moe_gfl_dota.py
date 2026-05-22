# 迁移自 mmdetection/local_configs/SARDet/SM3Det.py
# 
# 模型信息：
# - 原始Backbone: ConvNeXt_moe_MultiInput (多输入ConvNeXt with MoE)
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
#         pad_size_divisor=800
#     ),
#     backbone=dict(
#         type='ConvNeXt_moe_MultiInput',
#         MoE_Block_inds = [[],[],[],[]],
#         datasets=None,
#         arch='tiny',
#         drop_path_rate=0.1,
#     ),
#     neck=dict(
#         type='FPN',
#         in_channels=[96, 192, 384, 768],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_output',
#         num_outs=5),
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
        pad_size_divisor=800),
    # [迁移] ConvNeXt MoE backbone配置
    backbone=dict(
        type='ConvNeXt_moe_MultiInput',
        MoE_Block_inds=[[],[],[],[]],
        datasets=None,
        arch='tiny',
        drop_path_rate=0.1),
    # [保留] FPN neck配置
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
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

print("配置文件加载: mmrotate ConvNeXt-MoE GFL for SARDet (SM3Det)")
print("说明: 此配置基于mmdetection SARDet SM3Det模型迁移而来")
print("待操作: 1. 调整数据集路径和格式")
print("        2. 验证ConvNeXt_moe_MultiInput与mmrotate的兼容性")
print("        3. 根据实际旋转角度范围调整anchor_generator.angles配置")

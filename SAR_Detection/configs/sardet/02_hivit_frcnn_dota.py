# 迁移自 mmdetection/local_configs/SARDet/hivit_base_SARDet.py
# 
# 模型信息：
# - 原始Backbone: HiViT (分层Vision Transformer)
# - 原始Detector: FasterRCNN
# - 原始框架: mmdetection
# - 迁移框架: mmrotate (针对旋转目标检测调整)
#
# 改动说明：
# [原有设置已注释] 保留原有的mmdetection配置以供参考
# [新增设置] 添加mmrotate框架兼容的配置

# ========== 原始mmdetection配置 (已注释) ==========
# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     './_base_/default_runtime.py'
# ]
# pretrained = 'pretrained/mae_hivit_base_1600ep.pth'
# dataset_type = 'SAR_Det_Finegrained_Dataset'
# data_root = 'datasets/mmdetection/SARDet_100K'
# 
# model = dict(
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         bgr_to_rgb=False,
#         pad_size_divisor=16),
#     backbone=dict(
#         type='HiViT',
#         img_size=224,
#         patch_size=16,
#         embed_dim=512,
#         frozen_stages=-1,
#         depths=[2, 2, 20],
#         num_heads=8,
#         mlp_ratio=4.,
#         rpe=False,
#         drop_path_rate=0.0,
#         with_fpn=True,
#         out_indices=['H', 'M', 19, 19],
#         use_checkpoint=True,
#         global_indices=[4, 9, 14, 19],
#         window_size=14,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(
#         type='FPN',
#         in_channels=[128, 256, 512, 512],
#         out_channels=256,
#         num_outs=5),
#     roi_head=dict(
#         bbox_head=dict(
#             type='ConvFCBBoxHead',
#             num_shared_convs=4,
#             num_shared_fcs=1,
#             in_channels=256,
#             conv_out_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=6,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             reg_decoded_bbox=True,
#             norm_cfg=dict(type='SyncBN', requires_grad=True),
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='GIoULoss', loss_weight=10.0))),
# )

# ========== mmrotate新配置 ==========
_base_ = [
    '../_base_/models/rotated-faster-rcnn_r50_fpn.py',
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

num_classes = 6
pretrained = 'pretrained/mae_hivit_base_1600ep.pth'

# ========== 新backbone配置 ==========
model = dict(
    type='RotatedFasterRCNN',
    data_preprocessor=dict(
        type='RotatedDetDataPreprocessor',
        mean=[0, 0, 0],
        std=[1, 1, 1],
        bgr_to_rgb=False,
        pad_size_divisor=16),
    # [迁移] HiViT backbone配置
    backbone=dict(
        _delete_=True,
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        frozen_stages=-1,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=False,
        drop_path_rate=0.0,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=True,
        global_indices=[4, 9, 14, 19],
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    # [保留] FPN neck配置
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5),
    # [调整] roi_head配置 - mmrotate使用旋转检测head
    roi_head=dict(
        bbox_head=dict(
            type='RotatedBBoxHead',  # [改动] 从ConvFCBBoxHead改为RotatedBBoxHead
            num_classes=num_classes,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0., 0.],  # [改动] 增加角度维度
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),  # [改动] 增加角度维度
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
)

# ========== 数据集配置 [待调整] ==========
# [原始] 使用SARDet_100K数据集 (SAR检测)
# [新增] 需要调整为mmrotate格式的旋转目标检测数据集
# data_root = 'data/split_ss_dota/'  # 根据实际路径修改

print("配置文件加载: mmrotate HiViT Faster-RCNN for SARDet")
print("说明: 此配置基于mmdetection SARDet模型迁移而来")
print("待操作: 1. 调整数据集路径和格式")
print("        2. 确认HiViT预训练权重兼容性")

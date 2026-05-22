# 迁移自 mmdetection/local_configs/SARDet/r50_dota_pretrain/fg_frcnn_dota_pretrain_sar_wavelet_r50.py
# 
# 模型信息：
# - 原始Backbone: MSFA (多尺度特征适应) + SAR + Wavelet处理
# - 原始Detector: FasterRCNN
# - 原始框架: mmdetection
# - 迁移框架: mmrotate (针对旋转目标检测调整)
#
# 改动说明：
# [原有设置已注释] 保留原有的mmdetection配置以供参考
# [新增设置] 添加mmrotate框架兼容的配置

# ========== 原始mmdetection配置 ==========
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', 
    '../_base_/datasets/SARDet_100k.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

num_class = 6

# ========== 原始backbone配置 ==========
model = dict(
    init_cfg=dict(type='Pretrained', 
                  checkpoint='pretrained/fg_frcnn_dota_pretrain_sar_wavelet_r50/best_coco_bbox_mAP_epoch_12.pth'),
    backbone=dict(
        _delete_=True,
        type='MSFA',
        use_sar=True, 
        use_wavelet=True, 
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None
        ),
    ), 
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_class,
        )),
)

# ========== mmrotate新backbone配置 ==========
# model = dict(
#     type='RotatedFasterRCNN',
#     # [迁移] MSFA backbone配置
#     backbone=dict(
#         _delete_=True,
#         type='MSFA',
#         use_sar=True, 
#         use_wavelet=True,
#         input_size=(800, 800),
#         backbone=dict(
#             type='ResNet',
#             depth=50,
#             num_stages=4,
#             out_indices=(0, 1, 2, 3),
#             frozen_stages=1,
#             norm_cfg=dict(type='BN', requires_grad=True),
#             norm_eval=True,
#             style='pytorch'
#         ),
#     ),
#     # [保留] 原有detector设置
#     roi_head=dict(
#         bbox_head=dict(
#             num_classes=num_class,
#         )),
# )

# ========== 原始优化器设置 ==========
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        betas=(0.9, 0.999), 
        lr=0.0001, 
        type='AdamW', 
        weight_decay=0.05),
    type='OptimWrapper')

# ========== mmrotate新优化器设置 ==========
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         betas=(0.9, 0.999),
#         weight_decay=0.05),
#     type='OptimWrapper')

# ========== 数据集配置 [待调整] ==========
# 原始数据集：SARDet_100K (SAR检测数据集)
# mmrotate数据集：DOTA (旋转目标检测数据集)
# 注意：实际使用时需要根据具体数据集格式进行调整

# [注释] 原始数据加载配置
train_pipeline = [...]
test_pipeline = [...]
data_root = 'datasets/mmdetection/SARDet_100K/'


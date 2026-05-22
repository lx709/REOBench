_base_ = [
    './datasets/BigEarthNet-S2.py',
    # './schedules/ft.py',
    './schedules/schedule.py',
     './runtimes/default_runtime.py'
]

# ============================================================
# 模型配置
# ============================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PretrainedCROMA',
        # pretrained_path=None,
        pretrained_path='pretrained/CROMA_base.pt',
        size='base',
        modality='optical',
        image_resolution=120
    ),
    # neck=dict(type='LinearNeck', in_channels=768, out_channels=768),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
        # loss=dict(
        #     type='CrossEntropyLoss', 
        #     use_sigmoid=True,  # BCE loss for multi-label
        #     loss_weight=1.0
        # ),
        # loss=dict(
        #     type='AsymmetricLoss',   # 换成 AsymmetricLoss
        #     gamma_pos=0.0,           # 正样本 focusing 参数
        #     gamma_neg=4.0,           # 负样本 focusing 参数
        #     clip=0.05,               # optional probability margin
        #     loss_weight=1.0,
        #     use_sigmoid=True          # 保持多标签 BCE 风格
        # ),
        loss=dict(type='MultiLabelSoftMarginLoss', loss_weight=1.0),
    )
)
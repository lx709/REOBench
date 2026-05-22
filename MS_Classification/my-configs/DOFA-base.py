_base_ = [
    './datasets/BigEarthNet-S2.py',
    './schedules/schedule.py',
    # './schedules/ft.py',
    './runtimes/default_runtime.py'
]

# ============================================================
# 模型配置
# ============================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OFAViT',
        pretrained='pretrained/DOFA_ViT_base_e100.pth',
        # pretrained=None,
        img_size=120,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4),
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
    ))

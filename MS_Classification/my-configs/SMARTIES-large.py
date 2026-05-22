_base_ = [
    './datasets/BigEarthNet-S2.py',
    './schedules/schedule.py',
    './runtimes/default_runtime.py'
]

# ============================================================
# SMARTIES Vision Transformer Large Configuration for BigEarthNet-S2
# ============================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SmartiesViT',
        arch='large',  # ViT-Large: 1024 embed_dim, 24 layers, 16 heads
        img_size=120,
        patch_size=16,
        in_channels=12,  # BigEarthNet-S2 has 12 bands
        global_pool=False,
        mixed_precision='no',  # Options: 'no', 'fp16', 'bf16'
        pretrained=None,  # Path to pretrained weights if available
    ),
    neck=dict(type='LinearNeck', in_channels=1024, out_channels=1280),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,  # BigEarthNet has 19 classes
        in_channels=1280,
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,  # BCE loss for multi-label classification
            loss_weight=1.0
        ),
    )
)

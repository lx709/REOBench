_base_ = [
    './datasets/dfc2020_smarties.py',
    './runtimes/default_runtime.py',
    './schedules/schedule_frozenbackbone_epoch.py'  # SMARTIES-style schedule
]

# Model settings
norm_cfg = dict(type='BN', requires_grad=True)

# DFC2020 specific data preprocessor (no normalization for multispectral)
data_preprocessor = dict(
    type='MultispectralSegDataPreProcessor',
    mean=None,  # Normalization done in pipeline
    std=None,   # Normalization done in pipeline
    bgr_to_rgb=False,  # Multispectral data, not RGB
    pad_val=0,
    seg_pad_val=255,
    size=(96, 96))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SMARTIESViT',
        img_size=96,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),  # Output features from blocks 3, 5, 7, 11
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained='pretrained/smarties-v1-vitb-dfc2020-linprobe.safetensors',
        # spectrum_specs will use default DFC2020 configuration
    ),
    # No neck - SMARTIES style uses features directly
    decode_head=dict(
        type='SMARTIESSimpleHead',
        in_channels=768,           # ViT-Base embed_dim
        in_index=-1,               # Use last feature (from block 11)
        channels=768,              # Not used in SMARTIESSimpleHead, but required by base class
        num_classes=8,             # DFC2020 has 8 classes
        target_size=96,            # Target output size (DFC2020 uses 96x96)
        interpolate_mode='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',#LabelSmoothingCrossEntropy
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    # No auxiliary head - SMARTIES uses single head
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ============================================================================
# Training Configuration Notes
# ============================================================================
# This config uses SMARTIES-style training:
# - Frozen backbone (lr_mult=0.0 for backbone)
# - Only train the decode_head (simple 1x1 conv + upsample)
# - AdamW optimizer with lr=4e-3, betas=(0.9, 0.95), weight_decay=0.0
# - MultiStepLR: decay at epoch 60 and 80
# - 100 epochs total
# - Batch size should be adjusted based on your GPU memory
#   (SMARTIES uses 1024 for 96x96 patches, you may need smaller)

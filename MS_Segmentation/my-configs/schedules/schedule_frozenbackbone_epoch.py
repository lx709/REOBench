# AdamW optimizer with frozen backbone (Epoch-based)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,             # SMARTIES uses 1e-3 base lr, scaled to 4e-3 based on batch size
        betas=(0.9, 0.95),   # SMARTIES uses (0.9, 0.95) instead of (0.9, 0.999)
        weight_decay=0.0     # SMARTIES uses 0 weight decay for linear probing
    ),
    # Parameter-wise learning rate and weight decay configuration
    paramwise_cfg=dict(
        custom_keys={
            # ========== FREEZE BACKBONE ==========
            'backbone': dict(lr_mult=0.0),      # lr_mult=0.0 -> completely frozen

            # ========== NO WEIGHT DECAY ==========
            # These parameters should not use weight decay
            'pos_embed': dict(decay_mult=0.),   # Positional embeddings
            'cls_token': dict(decay_mult=0.),   # Class token
            'norm': dict(decay_mult=0.)          # All normalization layers
        }
    )
)

# Learning rate scheduler - MultiStepLR (SMARTIES style)
# Drops LR at 60% and 80% of total epochs
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[60, 80],  # Decay at epoch 60 and 80
        gamma=0.1             # Multiply LR by 0.1 at each milestone
    )
]

# ============================================================================
# Training Configuration
# ============================================================================

# Training loop (epoch-based)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,        # Total training epochs
    val_interval=1         # Validate every epoch
)

# Validation loop
val_cfg = dict(type='ValLoop')

# Test loop
test_cfg = dict(type='TestLoop')

# Hooks for training
default_hooks = dict(
    # Timer to measure iteration time
    timer=dict(type='IterTimerHook'),

    # Logger to print training info
    logger=dict(
        type='LoggerHook',
        interval=100,                    # Log every 50 iterations
        log_metric_by_epoch=False        # Log by epoch
    ),

    # Parameter scheduler hook (for LR scheduling)
    param_scheduler=dict(type='ParamSchedulerHook'),

    # Checkpoint saving
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,                  # Save by epoch
        interval=1,                     # Check every epoch
        save_best='mIoU',               # Save best model based on mIoU metric
        rule='greater',                 # Higher mIoU is better
        save_last=True,                 # Always save the latest checkpoint
        max_keep_ckpts=2                # Keep 2 checkpoints (last + best)
    ),

    # Sampler seed for distributed training
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # Visualization hook
    visualization=dict(type='SegVisualizationHook')
)

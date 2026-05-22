# AdamW optimizer with frozen backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,           # Higher LR than full fine-tuning (0.00006)
                             # Only training heads from scratch, can use larger LR
        betas=(0.9, 0.999),
        weight_decay=0.01
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

# Learning rate scheduler
param_scheduler = [
    # Warmup phase: 0 -> 1500 iterations
    # Gradually increase LR from very small value to base LR
    dict(
        type='LinearLR',
        start_factor=1e-6,  # Start from 1e-10 (1e-6 * 0.0001)
        by_epoch=False,
        begin=0,
        end=1500
    ),
    # Main training phase: 1500 -> 160000 iterations
    # Polynomial decay (power=1.0 means linear decay)
    dict(
        type='PolyLR',
        eta_min=0.0,        # Decay to 0
        power=1.0,          # Linear decay (power=1.0)
        begin=1500,
        end=160000,
        by_epoch=False
    )
]

# ============================================================================
# Training Configuration
# ============================================================================

# Training loop (iteration-based, not epoch-based)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,      # Total training iterations
    val_interval=8000     
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
        interval=100,                
        log_metric_by_epoch=False       # Log by iteration, not epoch
    ),

    # Parameter scheduler hook (for LR scheduling)
    param_scheduler=dict(type='ParamSchedulerHook'),

    # Checkpoint saving
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,                 # Save by iteration
        interval=16000,                 # Save every 16k iterations
        save_last=True,                 # Always save the latest checkpoint
        max_keep_ckpts=1                # Only keep 1 checkpoint (save disk space)
    ),

    # Sampler seed for distributed training
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # Visualization hook
    visualization=dict(type='SegVisualizationHook')
)
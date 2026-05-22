optimizer = dict(
    type='AdamW',
    lr=5e-3,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
)

# --------------------------
# scheduler
# --------------------------
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=15
    )
]

# --------------------------
# training schedule
# --------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --------------------------
# hooks
# --------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=2, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)
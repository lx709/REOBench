optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,                  # base learning rate
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
)

# ============================================================
# Learning Rate Scheduler
# Warmup (5 epochs) + Cosine Decay
# ============================================================
max_epochs = 50

param_scheduler = [
    # -------- Warmup --------
    dict(
        type='LinearLR',
        start_factor=1e-3,         # lr 从 0.1 * lr 开始
        by_epoch=True,
        begin=0,
        end=5
    ),
    # -------- Cosine Annealing --------
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        end=max_epochs,
        T_max=max_epochs - 5,
        eta_min=1e-7              # 比 1e-4 低两个数量级即可
    )
]

# ============================================================
# Training Config
# ============================================================
train_cfg = dict(
    by_epoch=True,
    max_epochs=max_epochs,
    val_interval=1
)

val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',         # Adam 默认参数
        lr=1e-4,             # 初始学习率
        betas=(0.9, 0.999),
        weight_decay=0.01      # linear probing 不使用 weight decay
    ),
    # 只优化 head 参数
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0)  # backbone 不更新
        }
    ),
)

# ============================================================
# 学习率调度器 - step decay at 60% and 80%
# ============================================================
# total_epochs = 100
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         milestones=[int(total_epochs*0.6), int(total_epochs*0.8)],
#         gamma=0.1,
#         by_epoch=True
#     )
# ]
total_epochs = 100
param_scheduler = [
    # -------- Warmup --------
    dict(
        type='LinearLR',
        start_factor=0.1,         # lr 从 0.1 * lr 开始
        by_epoch=True,
        begin=0,
        end=5
    ),
    # -------- Cosine Annealing --------
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        end=total_epochs,
        T_max=total_epochs - 5,
        eta_min=1e-6             # 比 1e-4 低两个数量级即可
    )
]

# ============================================================
# 训练配置
# ============================================================
train_cfg = dict(
    by_epoch=True,
    max_epochs=total_epochs,
    val_interval=1
)

# ============================================================
# 验证和测试配置
# ============================================================
val_cfg = dict()
test_cfg = dict()

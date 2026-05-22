default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=10,
    #     save_best="coco/bbox_mAP",
    #     rule="greater"),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                # 每个 epoch/迭代生成 checkpoint，但只保留下面指定的
        save_last=True,            # 保留最后一个 checkpoint
        save_best='coco/bbox_mAP', # 根据 metric 保存最好一个
        rule='greater',            # 评估 metric 越大越好
        max_keep_ckpts=2           # 最多保留两个 checkpoint
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

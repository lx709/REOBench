#!/bin/bash
#SBATCH --job-name=mmrotate-ft
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o sbatch-logs/randomrotate_lr_1e04_epoch36_%A_%a.out
#SBATCH --array=6
# 3 done
echo starting at $(date)
START_TIME=$(date +%s)
print_runtime() {
    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    printf 'Total runtime: %02d:%02d:%02d (exit code: %d)\n' \
        $((elapsed / 3600)) \
        $(((elapsed % 3600) / 60)) \
        $((elapsed % 60)) \
        "$exit_code"
}
trap print_runtime EXIT

CONFIGS=("configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_msfa_freeze_backbone_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_hivit_freeze_backbone_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_convnext_moe_freeze_backbone_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_r50_freeze_backbone_full.py" \
         "configs/sardet/rotated-gfl_r50_fpn_1x_sardet_from_gfl_pretrain.py" \
         "configs/sardet/rotated-gfl_fftresnet50_1x_sardet_from_denodet_pretrain.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarclip_vitb32_freeze_backbone_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarwmixmae_base_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarmae_vitb_full.py" \
         )


NAMES=("MSFA" "HIVIT" "SM3Det" "Grok" "Grok-RetinaNet" "Grok2-RetinaNet" "SARCLIP-ViTB32" "SARWMixMAE" "SARMAE-ViTB")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=randomrotate_lr_1e04_epoch36_${NAMES[$SLURM_ARRAY_TASK_ID]}
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    
echo "[INFO] 任务ID: $SLURM_ARRAY_TASK_ID"
echo "[INFO] 配置文件: $CONFIG"
echo "[INFO] 结果目录: $NAME"
echo "[INFO] 节点数: $NNODES"
echo "[INFO] 每节点GPU数: $GPUS_PER_NODE"
echo "[INFO] MASTER_ADDR: $MASTER_ADDR"
    
srun --export=ALL --ntasks=$NNODES --ntasks-per-node=1 singularity exec --nv --bind <YOUR_MMROTATE_DIR>:/mmrotate,<YOUR_PRETRAINED_DIR>:/mmrotate/pretrained,<YOUR_DATASETS_DIR>:/mmrotate/datasets <YOUR_SIF_IMAGE> bash -lc "
cd /mmrotate
export PYTHONUSERBASE=/mmrotate/.local
PYVER=$(python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=/mmrotate/.local/lib/python${PYVER}/site-packages:$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
echo [INFO] node_rank=\$SLURM_PROCID local_rank=\$SLURM_LOCALID host=\$(hostname) cuda_visible=\$CUDA_VISIBLE_DEVICES
NNODES=$NNODES NODE_RANK=\$SLURM_PROCID MASTER_ADDR=$MASTER_ADDR PORT=$((29570 + SLURM_ARRAY_TASK_ID)) ./tools/dist_train.sh $CONFIG $GPUS_PER_NODE --work-dir work_dirs/$NAME --cfg-options visualizer.vis_backends.1.init_kwargs.name=\"$NAME\"
"


# cd <YOUR_MMROTATE_DIR>; sbatch sbatch-jobs/sbatch-mmrotate-ft.sh;squeue
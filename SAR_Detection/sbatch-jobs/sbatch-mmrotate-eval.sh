#!/bin/bash

#SBATCH --job-name=mmrotate-eval
#SBATCH -o ./sbatch-logs/eval-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=4-5   # 4个SARDet模型 - 注意这个数字！

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
         "configs/sardet/rotated-gfl_r50_fpn_1x_sardet_from_gfl_pretrain.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarwmixmae_base_full.py" \
         "configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarmae_vitb_full.py"
         )

# [mmdetection权重] 
WORKDIRS=("work_dirs/mmrotate/randomrotate_lr_1e04_epoch24_MSFA/best_rcoco_bbox_mAP_epoch_22.pth" \
          "work_dirs/mmrotate/randomrotate_lr_1e04_epoch24_HIVIT/best_rcoco_bbox_mAP_epoch_19.pth" \
          "work_dirs/mmrotate/randomrotate_lr_1e04_epoch24_SM3Det/best_rcoco_bbox_mAP_epoch_24.pth" \
          "work_dirs/mmrotate/randomrotate_lr_1e04_epoch36_Grok-RetinaNet/best_rcoco_bbox_mAP_epoch_31.pth" \
          "work_dirs/randomrotate_lr_1e04_epoch36_SARWMixMAE/best_rcoco_bbox_mAP_epoch_36.pth"
          "work_dirs/randomrotate_lr_1e04_epoch36_SARMAE-ViTB/best_rcoco_bbox_mAP_epoch_31.pth"
          )
    

SIF_IMAGE=<YOUR_SIF_IMAGE>

# 获取当前数组任务对应的配置和权重
NAMES=("MSFA" "SARATR-X" "SM3Det" "Grok-RetinaNet" "SARWMixMAE")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
WORKDIR=${WORKDIRS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "[INFO] 任务ID: $SLURM_ARRAY_TASK_ID"
echo "[INFO] 配置文件: $CONFIG"
echo "[INFO] 权重文件: $WORKDIR"


AUG_TYPES=(rotate gaussian_blur scale gaps gaussian_noise translate rfi_interference speckle_noise)
# AUG_TYPES=(rotate)

SEVERITY=(1 2 3 4 5)


# ========== 容器环境配置 ==========
CONTAINER_CMD="
cd /mmrotate
export PYTHONUSERBASE=/mmrotate/.local
PYVER=\$(python3 - <<EOF
import sys
print(f'{sys.version_info.major}.{sys.version_info.minor}')
EOF
)
export PYTHONPATH=/mmrotate/.local/lib/python\${PYVER}/site-packages:\$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH
export WANDB_MODE=disabled
"

# export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

#              --show-dir visualize/ \ PORT=$((29500 + SLURM_ARRAY_TASK_ID)) ./tools/dist_test.sh

# ========== 生成数据增强测试命令 ==========
echo "[INFO] 开始生成评估命令..."

for aug in "${AUG_TYPES[@]}"; do
    for sev in "${SEVERITY[@]}"; do
        
        # [新增mmrotate兼容路径] - 根据实际数据集位置调整
        IMG_PATH="Corrupted_dota_style/val/images/${aug}/${sev}/"
        ANN_PATH="Corrupted_dota_style/val/annfiles/${aug}/${sev}/"
        
        # [配置选项] - 使用mmrotate的配置覆盖机制
        # 注意: mmrotate的配置选项格式可能与mmdetection略有不同
        CFG_OPTS="test_dataloader.dataset.data_prefix.img_path=${IMG_PATH} \
                  test_dataloader.dataset.ann_file=${ANN_PATH}"
        

           CMD="python tools/test.py $CONFIG \
               ${WORKDIR} \
             --cfg-options ${CFG_OPTS}; \
             echo '[DONE] Model: $NAME | aug=${aug} | sev=${sev} | img_path=${IMG_PATH}'"
        
        CONTAINER_CMD+="$CMD; "
    done
done

# ========== 原始数据（无增强）评估 ==========
# CMD="python tools/test.py ${CONFIG} \
#     ${WORKDIR} \
#     echo '[DONE] Model: $NAME | aug=origin (no corruption)'" 

# CONTAINER_CMD+="$CMD"

# ========== 执行评估 ==========
echo "[INFO] 启动Singularity容器进行评估..."


singularity exec --nv --cleanenv --bind <YOUR_MMROTATE_DIR>:/mmrotate,<YOUR_PRETRAINED_DIR>:/mmrotate/pretrained,<YOUR_DATASETS_DIR>:/mmrotate/datasets <YOUR_SIF_IMAGE> \
    /bin/bash -c "$CONTAINER_CMD"

echo "[INFO] 评估完成！结果已保存到对应目录。"

# cd <YOUR_MMROTATE_DIR>;sbatch sbatch-jobs/sbatch-mmrotate-eval.sh
#!/bin/bash

#SBATCH --job-name=mmrotate-eval-parallel
#SBATCH -o ./sbatch-logs/eval/eval-parallel-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --array=0-39   # 8 aug_types × 5 severities = 40 combinations

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

# ========== 固定模型：index 5，SARCLIP-ViTB32 ==========
CONFIG="configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarclip_vitb32_freeze_backbone_full.py"
WORKDIR="work_dirs/randomrotate_lr_1e04_epoch36_SARCLIP-ViTB32/best_rcoco_bbox_mAP_epoch_32.pth"
NAME="SARCLIP-ViTB32"

# ========== 从 array task ID 计算 aug / sev ==========
AUG_TYPES=(rotate gaussian_blur scale gaps gaussian_noise translate rfi_interference speckle_noise)
SEVERITY=(1 2 3 4 5)

AUG_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEV_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

AUG=${AUG_TYPES[$AUG_IDX]}
SEV=${SEVERITY[$SEV_IDX]}

echo "[INFO] 任务ID:    $SLURM_ARRAY_TASK_ID"
echo "[INFO] 模型:      $NAME"
echo "[INFO] aug_type:  $AUG  (idx=$AUG_IDX)"
echo "[INFO] severity:  $SEV  (idx=$SEV_IDX)"
echo "[INFO] 配置文件:  $CONFIG"
echo "[INFO] 权重文件:  $WORKDIR"

# ========== 路径 ==========
IMG_PATH="Corrupted_dota_style/val/images/${AUG}/${SEV}/"
ANN_PATH="Corrupted_dota_style/val/annfiles/${AUG}/${SEV}/"

CFG_OPTS="test_dataloader.dataset.data_prefix.img_path=${IMG_PATH} \
          test_dataloader.dataset.ann_file=${ANN_PATH}"

# ========== 容器内执行命令 ==========
CONTAINER_CMD="
cd /mmrotate
export PYTHONUSERBASE=/mmrotate/.local
PYVER=\$(python3 - <<EOF
import sys
print(f'{sys.version_info.major}.{sys.version_info.minor}')
EOF
)
export PYTHONPATH=/mmrotate/.local/lib/python\${PYVER}/site-packages:\$PYTHONPATH
export PATH=/mmrotate/.local/bin:\$PATH
export WANDB_MODE=disabled

python tools/test.py ${CONFIG} ${WORKDIR} --cfg-options ${CFG_OPTS}
echo '[DONE] Model: ${NAME} | aug=${AUG} | sev=${SEV}'
"

singularity exec --nv --cleanenv \
    --bind <YOUR_MMROTATE_DIR>:/mmrotate,<YOUR_PRETRAINED_DIR>:/mmrotate/pretrained,<YOUR_DATASETS_DIR>:/mmrotate/datasets \
    <YOUR_SIF_IMAGE> \
    /bin/bash -c "$CONTAINER_CMD"

# cd <YOUR_MMROTATE_DIR>; sbatch sbatch-jobs/sbatch-mmrotate-eval-parallel.sh

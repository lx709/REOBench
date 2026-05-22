#!/bin/bash
#SBATCH --job-name=mmptn-eval
#SBATCH --output=./sbatch-logs-eval/eval-%A_%a.out
#SBATCH --error=./sbatch-logs-eval/eval-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --array=0-3   

CONFIGS=("my-configs/CROMA-base.py" "my-configs/DOFA-base.py" \
         "my-configs/SMARTIES-base.py" "my-configs/dinov2-base.py")

WORKDIRS=("work_dirs/mmpretrain/CROMA-base/CROMA-base-lp-1e-4-256-stat-v1-19-MultiLabelSoftMarginLoss-testval-rotate/best_multi-label_mAP_epoch_90.pth" \
          "work_dirs/mmpretrain/DOFA-base/DOFA-base-lp-1e-4-256-stat-v1-19-MultiLabelSoftMarginLoss-testval-rotate/best_multi-label_mAP_epoch_99.pth" \
          "work_dirs/mmpretrain/SMARTIES-base/SMARTIES-base-lp-1e-4-256-percentile-v1-19-MultiLabelSoftMarginLoss-testval-rotate/best_multi-label_mAP_epoch_99.pth" \
          "work_dirs/mmpretrain/softcon-base/softcon-base-lp-1e-4-256-stat-8bit-v1-19-MultiLabelSoftMarginLoss-testval-rotate/best_multi-label_mAP_epoch_99.pth")

IMAGE=localhost/mmcv-pytorch:23.06

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
WORKDIR=${WORKDIRS[$SLURM_ARRAY_TASK_ID]}

# AUG_TYPES=(brightness_contrast compression_artifacts gaussian_blur haze rotate scale \
#            clouds gaps gaussian_noise motion_blur salt_and_pepper_noise translate)

AUG_TYPES=(brightness_contrast compression_artifacts gaussian_blur haze rotate scale \
           clouds gaps gaussian_noise motion_blur salt_and_pepper_noise translate)

SEVERITY=(1 2 3 4 5)

CONTAINER_CMD="export PYTHONUSERBASE=/workspace/.local; \
               PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
               export PYTHONPATH=/workspace/.local/lib/python${PYVER}/site-packages:$PYTHONPATH; \
               "

for aug in "${AUG_TYPES[@]}"; do
    for sev in "${SEVERITY[@]}"; do

        IMG_PATH="../BigEarthNet-S2-corrupted-val/${aug}/${sev}"
        CFG_OPTS="test_dataloader.dataset.data_prefix=${IMG_PATH}"

        CMD="python tools/test.py ${CONFIG} \
             ${WORKDIR} \
             --cfg-options ${CFG_OPTS}; \
             echo '[DONE] aug=${aug} sev=${sev} img_path=${IMG_PATH}'"

        CONTAINER_CMD+="$CMD; "
    done
done

podman-hpc run --rm \
    --shm-size=64g \
    --device=nvidia.com/gpu=all \
    ${IMAGE} /bin/bash -c "$CONTAINER_CMD"

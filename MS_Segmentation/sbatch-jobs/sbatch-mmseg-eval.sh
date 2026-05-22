#!/bin/bash
#SBATCH --job-name=mmseg-eval
#SBATCH --output=./sbatch-logs-eval/eval-%A_%a.out
#SBATCH --error=./sbatch-logs-eval/eval-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --array=0-3   # ################## BE ATTENTION!!!<<<<<<<<-----------------------------------------------------------------------------------

CONFIGS=("my-configs/CROMA-base.py" "my-configs/DOFA-base.py" \
         "my-configs/smarties-base.py" "my-configs/dinov2-base.py")

WORKDIRS=("work_dirs/CROMA-base/frozenbackbone-up100/best_mIoU_epoch_57.pth" \
          "work_dirs/dofa-base/frozenbackbone-up100/best_mIoU_epoch_19.pth" \
          "work_dirs/smarties-base/frozenbackbone-up100/best_mIoU_epoch_35.pth" \
          "work_dirs/softcon-base/frozenbackbone-lp100-64_1e_5/best_mIoU_epoch_49.pth")

IMAGE=localhost/mmcv-pytorch:23.06

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
WORKDIR=${WORKDIRS[$SLURM_ARRAY_TASK_ID]}

# AUG_TYPES=(brightness_contrast compression_artifacts gaussian_blur haze rotate scale \
#            clouds gaps gaussian_noise motion_blur salt_and_pepper_noise translate)
AUG_TYPES=(compression_artifacts-v2 rotate-v2)

SEVERITY=(1 2 3 4 5)

CONTAINER_CMD="export PYTHONUSERBASE=/workspace/.local; \
               PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
               export PYTHONPATH=/workspace/.local/lib/python${PYVER}/site-packages:$PYTHONPATH; \
               "
# 双层循环生成 test 命令
# for aug in "${AUG_TYPES[@]}"; do
#     for sev in "${SEVERITY[@]}"; do
#         CMD="python tools/test.py ${CONFIG} \
#              ${WORKDIR} \
#              --cfg-options test_dataloader.dataset.data_prefix.img_path='val/Corrupted_images/${aug}/${sev}'"
#         CONTAINER_CMD+="$CMD; "
#     done
# done
for aug in "${AUG_TYPES[@]}"; do
    for sev in "${SEVERITY[@]}"; do

        IMG_PATH="val/Corrupted_images/${aug}/${sev}"

        case "$aug" in
            rotate|scale|translate)
                ANN_PATH="val/Corrupted_labels/${aug}/${sev}"
                CFG_OPTS="test_dataloader.dataset.data_prefix.img_path=${IMG_PATH} \
                          test_dataloader.dataset.data_prefix.seg_map_path=${ANN_PATH}"
                ;;
            *)
                CFG_OPTS="test_dataloader.dataset.data_prefix.img_path=${IMG_PATH}"
                ;;
        esac

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
    -v <YOUR_WORKSPACE>:/workspace \
    -v <YOUR_DATASETS>:/datasets \
    -v <YOUR_PRETRAINED>:/workspace/pretrained \
    -v <YOUR_WORKDIRS>:/workspace/work_dirs \
    ${IMAGE} /bin/bash -c "$CONTAINER_CMD"

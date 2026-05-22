#!/bin/bash
#SBATCH --job-name=mmptn
#SBATCH --output=./sbatch-logs-train/train-%A_%a.out
#SBATCH --error=./sbatch-logs-train/train-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3   # 一次提交 4 个任务（索引 0,1,2,3）



CONFIGS=("my-configs/CROMA-base.py" "my-configs/DOFA-base.py" \
         "my-configs/SMARTIES-base.py" "my-configs/dinov2-base.py")

MODE="lp"
LR="1e-4"
BATCH="256"
NORM="stat"
FULL="v1-19-MultiLabelSoftMarginLoss-rotate"
WORKDIRS=("work_dirs/CROMA-base/CROMA-base-${MODE}-${LR}-${BATCH}-${NORM}-${FULL}" \
          "work_dirs/DOFA-base/DOFA-base-${MODE}-${LR}-${BATCH}-${NORM}-${FULL}" \
          "work_dirs/SMARTIES-base/SMARTIES-base-${MODE}-${LR}-${BATCH}-percentile-${FULL}" \
          "work_dirs/softcon-base/softcon-base-${MODE}-${LR}-${BATCH}-${NORM}-8bit-${FULL}")

IMAGE=localhost/mmcv-pytorch:23.06

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
WORKDIR=${WORKDIRS[$SLURM_ARRAY_TASK_ID]}

podman-hpc run --rm \
    --shm-size=64g \
    --device=nvidia.com/gpu=all \
    ${IMAGE} /bin/bash -c export PYTHONPATH=/workspace/.local/lib/python3.10/site-packages:\$PYTHONPATH; python tools/train.py ${CONFIG} --work-dir ${WORKDIR}"

#!/bin/bash
#SBATCH --job-name=mmseg
#SBATCH --output=./sbatch-logs/hyper-parameters-%t-%j.out
#SBATCH --error=./sbatch-logs/hyper-parameters-%t-%j.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=10:00:00

# 定义容器卷映射和镜像
WORKSPACE=<YOUR_WORKSPACE>          # e.g. /home/<user>/mmsegmentation
DATASETS=<YOUR_DATASETS>            # e.g. /projects/<proj>/datasets/DFC2020
PRETRAINED=<YOUR_PRETRAINED>        # e.g. /projects/<proj>/pretrained
WORKDIRS=<YOUR_WORKDIRS>            # e.g. /projects/<proj>/work_dirs/mmsegmentation
IMAGE=localhost/mmcv-pytorch:23.06



CONTAINER_CMD=$(cat <<'EOF'
export PYTHONUSERBASE=/workspace/.local
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=/workspace/.local/lib/python${PYVER}/site-packages:$PYTHONPATH
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
# export WANDB_DISABLED=true
EOF
)

podman-hpc run --rm \
    --shm-size=64g \
    --device=nvidia.com/gpu=all \
    -v ${WORKSPACE}:/workspace \
    -v ${DATASETS}:/datasets \
    -v ${PRETRAINED}:/workspace/pretrained \
    -v ${WORKDIRS}:/workspace/work_dirs \
    ${IMAGE} /bin/bash -c "${CONTAINER_CMD}"
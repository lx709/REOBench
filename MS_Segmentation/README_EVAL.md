# MS_Segmentation — Setup & Evaluation Guide

This module uses the **mmsegmentation** framework for semantic segmentation evaluation on multispectral remote sensing images.

---

## 1. Environment Setup

Follow the [mmsegmentation official documentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) to install dependencies.

```bash
# 1. Install PyTorch (choose the version matching your CUDA)
pip install torch torchvision

# 2. Install mmengine and mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# 3. Install mmsegmentation in editable mode from this directory
pip install -e .
```

> Alternatively, run inside the container image `localhost/mmcv-pytorch:23.06` (podman-hpc). See `sbatch-jobs/sbatch-mmseg-eval.sh` for reference.
> Environment variable setup: refer to `env.sh` or `pytorch_conda_env.yaml`.

---

## 2. Fine-tuning

The training script is at `sbatch-jobs/sbatch-mmseg-train.sh`. It trains a single model per job submission.

Before submitting, set the volume paths at the top of the script:

```bash
WORKSPACE=<YOUR_WORKSPACE>    # e.g. /home/<user>/mmsegmentation
DATASETS=<YOUR_DATASETS>      # e.g. /projects/<proj>/datasets/DFC2020
PRETRAINED=<YOUR_PRETRAINED>  # e.g. /projects/<proj>/pretrained
WORKDIRS=<YOUR_WORKDIRS>      # e.g. /projects/<proj>/work_dirs/mmsegmentation
```

Also set your WANDB API key inside the script:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

Then submit with:

```bash
cd MS_Segmentation
sbatch sbatch-jobs/sbatch-mmseg-train.sh
```

Edit the `WANDB_NAME` and `python tools/train.py` command inside the script to select the model and config to train. Example:

```bash
WANDB_NAME="seg-softcon-base-lp100-64_1e_5" \
    python tools/train.py my-configs/dinov2-base.py \
    --work-dir work_dirs/softcon-base/frozenbackbone-lp100-64_1e_5
```

Training logs are written to `sbatch-logs/`.

---

## 3. Evaluation Script

The evaluation script is at `sbatch-jobs/sbatch-mmseg-eval.sh`. It uses a SLURM array job to evaluate 4 models in parallel.

Submit with:

```bash
cd MS_Segmentation
sbatch sbatch-jobs/sbatch-mmseg-eval.sh
```

---

## 4. Four Evaluation Configurations

| # | Model | Config |
|---|-------|--------|
| 0 | CROMA-base | `my-configs/CROMA-base.py` |
| 1 | DOFA-base | `my-configs/DOFA-base.py` |
| 2 | SMARTIES-base | `my-configs/smarties-base.py` |
| 3 | SoftCon-base (DINOv2) | `my-configs/dinov2-base.py` |

To run a single model manually:

```bash
python tools/test.py my-configs/CROMA-base.py <YOUR_CHECKPOINT>
```

---

## 5. Corruption Evaluation

The evaluation script iterates over the following corruption types x 5 severity levels (currently active):

```
compression_artifacts-v2   rotate-v2
```

Logs are written to `sbatch-logs-eval/`.

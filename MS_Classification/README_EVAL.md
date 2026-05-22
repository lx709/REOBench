# MS_Classification — Setup & Evaluation Guide

This module uses the **mmpretrain** framework for multi-label classification evaluation on multispectral remote sensing images.

---

## 1. Environment Setup

Follow the [mmpretrain official documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) to install dependencies.

```bash
# 1. Install PyTorch (choose the version matching your CUDA)
pip install torch torchvision

# 2. Install mmengine and mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# 3. Install mmpretrain in editable mode from this directory
pip install -e .
```

> Alternatively, run inside the container image `localhost/mmcv-pytorch:23.06` (podman-hpc). See `sbatch-jobs/sbatch-mmpretrain-eval.sh` for reference.

---

## 2. Fine-tuning

The training script is at `sbatch-jobs/sbatch-mmpretrain-train.sh`. It submits 4 models as a SLURM array job.

Submit with:

```bash
cd MS_Classification
sbatch sbatch-jobs/sbatch-mmpretrain-train.sh
```

The four training configurations share these hyperparameters:

| Parameter | Value |
|-----------|-------|
| Mode | `lp` (linear probing) |
| Learning rate | `1e-4` |
| Batch size | `256` |
| Normalization | `stat` (SMARTIES uses `percentile`) |
| Time limit | 24 h / job |

| # | Model | Config | Output dir |
|---|-------|--------|------------|
| 0 | CROMA-base | `my-configs/CROMA-base.py` | `work_dirs/CROMA-base/CROMA-base-lp-1e-4-256-stat-v1-19-MultiLabelSoftMarginLoss-rotate/` |
| 1 | DOFA-base | `my-configs/DOFA-base.py` | `work_dirs/DOFA-base/DOFA-base-lp-1e-4-256-stat-v1-19-MultiLabelSoftMarginLoss-rotate/` |
| 2 | SMARTIES-base | `my-configs/SMARTIES-base.py` | `work_dirs/SMARTIES-base/SMARTIES-base-lp-1e-4-256-percentile-v1-19-MultiLabelSoftMarginLoss-rotate/` |
| 3 | SoftCon-base (DINOv2) | `my-configs/dinov2-base.py` | `work_dirs/softcon-base/softcon-base-lp-1e-4-256-stat-8bit-v1-19-MultiLabelSoftMarginLoss-rotate/` |

Training logs are written to `sbatch-logs-train/`.

---

## 3. Evaluation Script

The evaluation script is at `sbatch-jobs/sbatch-mmpretrain-eval.sh`. It uses a SLURM array job to evaluate 4 models in parallel.

Submit with:

```bash
cd MS_Classification
sbatch sbatch-jobs/sbatch-mmpretrain-eval.sh
```

---

## 4. Four Evaluation Configurations

| # | Model | Config |
|---|-------|--------|
| 0 | CROMA-base | `my-configs/CROMA-base.py` |
| 1 | DOFA-base | `my-configs/DOFA-base.py` |
| 2 | SMARTIES-base | `my-configs/SMARTIES-base.py` |
| 3 | SoftCon-base (DINOv2) | `my-configs/dinov2-base.py` |

To run a single model manually:

```bash
python tools/test.py my-configs/CROMA-base.py <YOUR_CHECKPOINT>
```

---

## 5. Corruption Evaluation

The evaluation script iterates over 12 corruption types x 5 severity levels:

```
brightness_contrast  compression_artifacts  gaussian_blur  haze
rotate               scale                  clouds         gaps
gaussian_noise       motion_blur            salt_and_pepper_noise  translate
```

Logs are written to `sbatch-logs-eval/`.

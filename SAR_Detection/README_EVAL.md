# SAR_Detection — Setup & Evaluation Guide

This module uses the **mmrotate** framework for oriented object detection evaluation on SAR images.

---

## 1. Environment Setup

Follow the [mmrotate official documentation](https://mmrotate.readthedocs.io/en/latest/get_started.html) to install dependencies.

```bash
# 1. Install PyTorch (choose the version matching your CUDA)
pip install torch torchvision

# 2. Install mmengine, mmcv, and mmdet
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"
mim install mmdet

# 3. Install mmrotate in editable mode from this directory
pip install -e .
```

> See `env_constraints.txt` for exact version constraints.

---

## 2. Fine-tuning

The training script is at `sbatch-jobs/sbatch-mmrotate-ft.sh`. It runs multi-node distributed training via Singularity + SLURM.

Submit with:

```bash
cd SAR_Detection
# set --array to the index of the model you want to train (see table below)
sbatch sbatch-jobs/sbatch-mmrotate-ft.sh
```

**SLURM resources per job:** 2 nodes × 4 GPUs, 24 h time limit.

Before submitting, set the following placeholders in the script:

```bash
# Singularity bind mounts
--bind <YOUR_MMROTATE_DIR>:/mmrotate,<YOUR_PRETRAINED_DIR>:/mmrotate/pretrained,<YOUR_DATASETS_DIR>:/mmrotate/datasets

# SIF image
$HOME/sif-images/pytorch_23.06.sif   # adjust path if needed

# WANDB
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

The three active training configurations:

| Array index | Model | Config |
|-------------|-------|--------|
| 1 | HiViT | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_hivit_freeze_backbone_full.py` |
| 7 | SAR-W-MixMAE | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarwmixmae_base_full.py` |
| 8 | SARMAE-ViTB | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarmae_vitb_full.py` |

Output is written to `work_dirs/randomrotate_lr_1e04_epoch36_<MODEL_NAME>/`.

---

## 3. Evaluation Scripts

This module has no `sbatch-jobs/` directory. Use the generic scripts under `tools/`.

**Single-GPU evaluation:**

```bash
python tools/test.py <CONFIG> <CHECKPOINT>
```

**Multi-GPU distributed evaluation:**

```bash
bash tools/dist_test.sh <CONFIG> <CHECKPOINT> <GPU_NUM>
```

**SLURM cluster evaluation:**

```bash
bash tools/slurm_test.sh <PARTITION> <JOB_NAME> <CONFIG> <CHECKPOINT>
```

---

## 4. Four Main Evaluation Configurations (SARDet Dataset)

Config files are located in `configs/sardet/`:

| # | Model | Config |
|---|-------|--------|
| 1 | MSFA + R50 + FPN | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_msfa_freeze_backbone_full.py` |
| 2 | HiViT + FPN | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_hivit_freeze_backbone_full.py` |
| 3 | SARClip ViT-B/16 + FPN | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarclip_vitb16_freeze_backbone_full.py` |
| 4 | SARMAE ViT-B + FPN | `configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_sarmae_vitb_full.py` |

Example run:

```bash
python tools/test.py \
    configs/sardet/oriented-rcnn-le90_r50_fpn_1x_sardet_msfa_freeze_backbone_full.py \
    <CHECKPOINT_PATH>
```

Additional configs (ConvNeXt-MoE, DINOv2, SAR-W-MixMAE variants, etc.) are available in `configs/sardet/`.

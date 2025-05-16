This folder is based on [mmrotate](https://github.com/open-mmlab/mmrotate)

## Installation
```
conda create -n mmrotate python=3.7 -y
conda activate mmrotate
module load CUDA/11.4.1
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
pip install timm==0.6.13
pip install ftfy
pip install regex
pip install einops
pip install open-clip-torch
```

## Get Started

Configuration files for all benchmark models are stored in ```configs/DIOR``` folder. Check ```docs/en/tutorials/customize_dataset.md``` for instructions on how to train and test a model with customized configs.

Training Oriented R-CNN with ScaleMAE backbone on DIOR dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/DIOR/scaleMAE.py
```

EVALUATE Oriented R-CNN with ScaleMAE backbone on DIOR dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/DIOR/EVAL/scaleMAE/{clouds}/{1}.py
```

## References

The codes are mainly borrowed from [mmrotate](https://github.com/open-mmlab/mmrotate) and [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA).

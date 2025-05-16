### Install

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation

### Training & Evaluation

Configuration files for all benchmark models are stored in ```configs/vit_base_win``` folder. Check ```docs/en/train.md``` and ```docs/en/inference.md``` for instructions on how to train and test a model with customized configs.

Training UperNet with ScaleMAE backbone on Potsdam dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/vit_base_win/scale_MAE.py --launcher 'pytorch'
```

Evaluate UperNet with ScaleMAE backbone on Potsdam dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40002 tools/test.py \
configs/vit_base_win/scale_MAE_corrupted.py --launcher 'pytorch'
```

## References

The codes are mainly borrowed from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA).
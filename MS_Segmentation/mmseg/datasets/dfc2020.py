"""
DFC2020 Multispectral Dataset for MMSegmentation
Handles 14-band multispectral data (Sentinel-2 + Sentinel-1)
"""
import os.path as osp
import numpy as np
import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DFC2020Dataset(BaseSegDataset):
    """DFC2020 dataset for multispectral semantic segmentation.

    The dataset contains 14 bands:
    - 12 Sentinel-2 bands (optical/infrared)
    - 2 Sentinel-1 bands (SAR)

    Args:
        img_suffix (str): Suffix of images. Default: '.tif'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.tif'
        reduce_zero_label (bool): Whether to mark label zero as ignored. Default: False
    """

    METAINFO = dict(
        classes=('Forest', 'Shrubland', 'Grassland', 'Wetlands',
                'Croplands', 'Urban', 'Barren', 'Water'),
        palette=[[0, 100, 0], [150, 250, 0], [255, 255, 100], [0, 150, 150],
                [255, 200, 0], [200, 0, 0], [150, 150, 150], [0, 0, 255]]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs):
        super(DFC2020Dataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

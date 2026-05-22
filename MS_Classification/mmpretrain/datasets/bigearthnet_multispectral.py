# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
from typing import List
import numpy as np
import rasterio
import torch

from mmpretrain.registry import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class BigEarthNetMultiSpectral(MultiLabelDataset):
    """BigEarthNet Multi-Spectral Dataset for multi-label classification.

    This dataset supports loading multi-spectral satellite imagery from BigEarthNet-S2.
    Unlike standard RGB datasets, this handles 12-channel Sentinel-2 data.

    Args:
        data_root (str): Root directory of the dataset.
        ann_file (str): Annotation file path (not used, data loaded from folder structure).
        pipeline (list): Processing pipeline.
        bands (str): Which bands to load - 's1', 's2', 'all', or 'rgb'. Default: 's2'.
        num_classes (int): Number of classes. Default: 19.
        **kwargs: Other arguments passed to MultiLabelDataset.
    """

    # BigEarthNet-v2 19 classes
    METAINFO = {
        'classes': [
            'Agro-forestry areas', 'Airports',
            'Annual crops associated with permanent crops', 'Bare rock',
            'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
            'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
            'Construction sites', 'Continuous urban fabric',
            'Discontinuous urban fabric', 'Green urban areas',
            'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
            'Land principally occupied by agriculture, with significant areas of '
            'natural vegetation', 'Marine waters'
        ]
    }

    def __init__(self,
                 data_root,
                 ann_file='',
                 pipeline=(),
                 bands='s2',
                 num_classes=19,
                 **kwargs):
        self.bands = bands
        self.num_classes = num_classes

        # For BigEarthNet, we don't use ann_file, we scan the directory
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            **kwargs
        )

    def load_data_list(self) -> List[dict]:
        """Load data list from BigEarthNet directory structure."""
        data_list = []

        # BigEarthNet directory structure: data_root/tile_id/patch_id/
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root {self.data_root} not found")

        # Find all patch directories
        patch_dirs = []
        for tile_dir in os.listdir(self.data_root):
            tile_path = os.path.join(self.data_root, tile_dir)
            if os.path.isdir(tile_path):
                for patch_dir in os.listdir(tile_path):
                    patch_path = os.path.join(tile_path, patch_dir)
                    if os.path.isdir(patch_path):
                        patch_dirs.append(patch_path)

        for patch_dir in patch_dirs:
            # Load band files
            band_files = sorted(glob.glob(os.path.join(patch_dir, '*.tif')))
            if not band_files:
                continue

            # Load labels from metadata (if exists)
            labels_file = os.path.join(patch_dir, f'{os.path.basename(patch_dir)}_labels_metadata.json')
            gt_label = self._load_labels(labels_file) if os.path.exists(labels_file) else []

            data_info = {
                'img_path': patch_dir,
                'gt_label': np.array(gt_label),
                'band_files': band_files,
            }
            data_list.append(data_info)

        return data_list

    def _load_labels(self, labels_file):
        """Load labels from JSON metadata file."""
        import json
        with open(labels_file, 'r') as f:
            metadata = json.load(f)

        label_names = metadata.get('labels', [])
        # Convert label names to indices
        gt_label = []
        for label_name in label_names:
            if label_name in self.METAINFO['classes']:
                gt_label.append(self.METAINFO['classes'].index(label_name))

        return gt_label

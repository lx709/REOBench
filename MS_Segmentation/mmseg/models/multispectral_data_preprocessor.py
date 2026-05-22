"""
Custom data preprocessor for multispectral segmentation with SMARTIES
"""
import torch
from typing import Dict, List, Optional, Union
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class MultispectralSegDataPreProcessor(BaseDataPreprocessor):
    """Data preprocessor for multispectral semantic segmentation.

    This preprocessor handles the projection indices needed by SMARTIES model.

    Args:
        mean (Sequence[float]): Mean values for normalization (applied in pipeline).
        std (Sequence[float]): Std values for normalization (applied in pipeline).
        size (tuple): Expected input size (H, W).
        bgr_to_rgb (bool): Whether to convert BGR to RGB (not used for multispectral).
        pad_val (float): Padding value for images.
        seg_pad_val (int): Padding value for segmentation maps.
    """

    def __init__(self,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 size: Optional[tuple] = None,
                 bgr_to_rgb: bool = False,
                 pad_val: float = 0,
                 seg_pad_val: int = 255,
                 **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # Note: normalization is handled in the pipeline for multispectral data
        # mean/std here are just placeholders for compatibility

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Forward function.

        Args:
            data (dict): Data from dataloader. Contains:
                - inputs: image tensor (already normalized in pipeline)
                - proj_indices: projection indices for SMARTIES
                - data_samples: contains gt_sem_seg and img_meta
            training (bool): Whether in training mode.

        Returns:
            dict: Preprocessed data with:
                - inputs: tuple of (imgs, proj_indices) for SMARTIES
                - data_samples: ground truth and metadata
        """
        # Get inputs and move to device
        inputs = data['inputs']
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        inputs = inputs.to(self.device)

        # Get projection indices (same for all samples in batch)
        if 'proj_indices' in data:
            proj_indices = data['proj_indices']
            if isinstance(proj_indices, list):
                # All samples should have same proj_indices
                proj_indices = proj_indices[0]
            proj_indices = proj_indices.to(self.device)
        else:
            # Fallback: shouldn't happen if pipeline is correct
            raise ValueError("proj_indices not found in data batch")

        # Process data samples (ground truth, metadata, etc.)
        data_samples = data.get('data_samples', None)

        # Move ground truth to device
        if data_samples is not None:
            if isinstance(data_samples, list):
                for data_sample in data_samples:
                    if hasattr(data_sample, 'gt_sem_seg'):
                        data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.data.to(self.device)
            else:
                if hasattr(data_samples, 'gt_sem_seg'):
                    data_samples.gt_sem_seg.data = data_samples.gt_sem_seg.data.to(self.device)

        # Return in format expected by SMARTIES model
        # inputs is a tuple (imgs, proj_indices)
        processed_data = {
            'inputs': (inputs, proj_indices),
            'data_samples': data_samples
        }

        return processed_data

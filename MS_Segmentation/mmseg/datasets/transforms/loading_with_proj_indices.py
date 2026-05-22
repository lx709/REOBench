# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom transforms for loading multi-spectral remote sensing images with projection indices.
This module supports loading Sentinel-1/Sentinel-2 and other multi-band imagery with
spectrum-aware projection indices for SMARTIES-style models.
"""

from typing import Dict, Optional, List
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class LoadMultiSpectralImageWithProjIndices(BaseTransform):
    """Load multi-spectral remote sensing image with projection indices.

    This transform loads multi-band tif files and generates projection indices
    that indicate which spectrum projection to use for each band. This is designed
    for SMARTIES-style models that use spectrum-aware projections.

    Required Keys:
    - img_path (str): Path to the multi-band image file

    Added Keys:
    - img (np.ndarray): Multi-band image with shape (H, W, C)
    - proj_indices (np.ndarray): Projection indices for each band, shape (C,)
    - img_shape (tuple): Image shape (H, W)
    - ori_shape (tuple): Original image shape (H, W)
    - num_bands (int): Number of spectral bands

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to True.
        sensor_specs (dict, optional): Sensor specifications defining bands and
            projection indices. If None, will be inferred from metadata or defaults.
            Example structure:
            {
                'sentinel2': {
                    'sensor_idx': 0,
                    'bands': ['B01', 'B02', ..., 'B12'],
                    'selected_bands': [0, 1, 2, ...],  # indices to use
                },
                'sentinel1': {
                    'sensor_idx': 1,
                    'bands': ['VV', 'VH'],
                    'selected_bands': [0, 1],
                }
            }
        spectrum_specs (dict, optional): Spectrum specifications defining projection
            indices for each band. If None, defaults will be used.
            Example structure:
            {
                'B01': {'projection_idx': 0, 'agg_projections': []},
                'B02': {'projection_idx': 1, 'agg_projections': []},
                ...
            }
        auto_detect_sensor (bool): Whether to auto-detect sensor type based on
            number of bands. Defaults to True.
    """

    # Default projection mapping for common sensors
    DEFAULT_SENSOR_MAPPING = {
        3: 'rgb',          # RGB: 3 bands
        12: 'sentinel2',   # Sentinel-2: 12 bands (without B10)
        13: 'sentinel2_full',  # Sentinel-2: 13 bands
        2: 'sentinel1',    # Sentinel-1: 2 bands (VV, VH)
        4: 'rgbn',         # RGB + NIR: 4 bands
    }

    def __init__(
        self,
        to_float32: bool = True,
        sensor_specs: Optional[Dict] = None,
        spectrum_specs: Optional[Dict] = None,
        auto_detect_sensor: bool = True,
        default_projection_idx: int = 0,
    ):
        if gdal is None:
            raise RuntimeError('gdal is not installed. Please install it via: '
                             'pip install gdal or conda install gdal')

        self.to_float32 = to_float32
        self.sensor_specs = sensor_specs
        self.spectrum_specs = spectrum_specs
        self.auto_detect_sensor = auto_detect_sensor
        self.default_projection_idx = default_projection_idx

        # Initialize projection conversion if spectrum_specs is provided
        self.projection_conversion = None
        if spectrum_specs is not None:
            self.projection_conversion = {
                band: spec['projection_idx']
                for band, spec in spectrum_specs.items()
            }

    def _get_default_proj_indices(self, num_bands: int, sensor_type: str = None) -> np.ndarray:
        """Generate default projection indices based on sensor type.

        Args:
            num_bands (int): Number of bands in the image
            sensor_type (str, optional): Type of sensor

        Returns:
            np.ndarray: Projection indices for each band
        """
        if sensor_type == 'rgb' or num_bands == 3:
            # RGB: bands typically map to projection indices 0, 1, 2
            return np.array([0, 1, 2])

        elif sensor_type in ['sentinel2', 'sentinel2_full'] or num_bands in [12, 13]:
            # Sentinel-2: 12-13 bands with different projection indices
            # You can customize this based on your spectrum_specs
            if num_bands == 12:
                return np.arange(12)  # projection_idx 0-11
            else:
                return np.arange(13)  # projection_idx 0-12

        elif sensor_type == 'sentinel1' or num_bands == 2:
            # Sentinel-1: VV, VH typically use separate projections
            return np.array([0, 1])

        elif sensor_type == 'rgbn' or num_bands == 4:
            # RGB + NIR
            return np.array([0, 1, 2, 3])

        else:
            # Default: sequential projection indices
            return np.arange(num_bands)

    def _detect_sensor_type(self, num_bands: int) -> Optional[str]:
        """Auto-detect sensor type based on number of bands.

        Args:
            num_bands (int): Number of bands

        Returns:
            str or None: Detected sensor type
        """
        return self.DEFAULT_SENSOR_MAPPING.get(num_bands, None)

    def transform(self, results: Dict) -> Dict:
        """Transform function to load multi-spectral image with projection indices.

        Args:
            results (dict): Result dict from dataset.

        Returns:
            dict: Updated result dict with image and projection indices.
        """
        filename = results['img_path']

        # Load image using GDAL
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')

        # Read all bands: shape (C, H, W) -> transpose to (H, W, C)
        img = np.einsum('ijk->jki', ds.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)

        # Get number of bands
        num_bands = img.shape[2] if len(img.shape) == 3 else 1

        # Auto-detect sensor type if enabled
        sensor_type = None
        if self.auto_detect_sensor:
            sensor_type = self._detect_sensor_type(num_bands)

        # Generate projection indices
        if self.sensor_specs is not None and self.spectrum_specs is not None:
            # Use provided specifications
            proj_indices = self._generate_proj_indices_from_specs(num_bands)
        else:
            # Use default projection indices
            proj_indices = self._get_default_proj_indices(num_bands, sensor_type)

        # Update results
        results['img'] = img
        results['proj_indices'] = proj_indices.astype(np.int32)
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['num_bands'] = num_bands

        # Store sensor type if detected
        if sensor_type is not None:
            results['sensor_type'] = sensor_type

        return results

    def _generate_proj_indices_from_specs(self, num_bands: int) -> np.ndarray:
        """Generate projection indices from sensor and spectrum specs.

        Args:
            num_bands (int): Number of bands

        Returns:
            np.ndarray: Projection indices
        """
        # This is a simplified version - you can extend it based on your needs
        if self.projection_conversion is not None:
            # Use the projection conversion mapping
            proj_indices = []
            for i in range(num_bands):
                # Map band index to projection index
                # This assumes bands are ordered as in spectrum_specs
                band_name = list(self.projection_conversion.keys())[i] if i < len(self.projection_conversion) else None
                if band_name:
                    proj_indices.append(self.projection_conversion[band_name])
                else:
                    proj_indices.append(self.default_projection_idx)
            return np.array(proj_indices)
        else:
            return np.arange(num_bands)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                   f'to_float32={self.to_float32}, '
                   f'auto_detect_sensor={self.auto_detect_sensor})')
        return repr_str


@TRANSFORMS.register_module()
class AddProjectionIndices(BaseTransform):
    """Add projection indices to existing image data.

    This transform is useful when you've already loaded the image using
    another transform (e.g., LoadSingleRSImageFromFile) and just want to
    add projection indices.

    Required Keys:
    - img (np.ndarray): Image with shape (H, W, C)

    Added Keys:
    - proj_indices (np.ndarray): Projection indices for each band

    Args:
        proj_indices (list or np.ndarray, optional): Manual projection indices.
            If None, will be auto-generated based on number of bands.
        sensor_type (str, optional): Sensor type for default projection indices.
            Options: 'rgb', 'sentinel1', 'sentinel2', 'rgbn', etc.
    """

    def __init__(
        self,
        proj_indices: Optional[List[int]] = None,
        sensor_type: Optional[str] = None,
    ):
        self.manual_proj_indices = proj_indices
        self.sensor_type = sensor_type

    def transform(self, results: Dict) -> Dict:
        """Add projection indices to results.

        Args:
            results (dict): Result dict with image.

        Returns:
            dict: Updated result dict with projection indices.
        """
        img = results['img']
        num_bands = img.shape[2] if len(img.shape) == 3 else 1

        if self.manual_proj_indices is not None:
            # Use manually specified projection indices
            proj_indices = np.array(self.manual_proj_indices, dtype=np.int32)
            if len(proj_indices) != num_bands:
                raise ValueError(
                    f'Number of projection indices ({len(proj_indices)}) '
                    f'does not match number of bands ({num_bands})'
                )
        else:
            # Auto-generate based on sensor type or default
            proj_indices = self._get_default_proj_indices(num_bands)

        results['proj_indices'] = proj_indices
        results['num_bands'] = num_bands

        return results

    def _get_default_proj_indices(self, num_bands: int) -> np.ndarray:
        """Generate default projection indices."""
        sensor_mapping = {
            'rgb': lambda: np.array([0, 1, 2]),
            'sentinel1': lambda: np.array([0, 1]),
            'sentinel2': lambda: np.arange(12),
            'rgbn': lambda: np.array([0, 1, 2, 3]),
        }

        if self.sensor_type in sensor_mapping:
            return sensor_mapping[self.sensor_type]()
        else:
            # Default: sequential indices
            return np.arange(num_bands, dtype=np.int32)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                   f'proj_indices={self.manual_proj_indices}, '
                   f'sensor_type={self.sensor_type})')
        return repr_str

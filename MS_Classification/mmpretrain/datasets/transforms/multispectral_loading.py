# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import numpy as np
import tifffile
from mmcv.transforms import BaseTransform

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultispectralImageFromFile(BaseTransform):
    """Load multispectral image from TIFF file using tifffile library.

    This transform loads multi-band TIFF images (e.g., Sentinel-2 imagery)
    and converts them to numpy arrays. It's designed to replace rasterio-based
    loading for simpler environments.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to True.
        channel_first (bool): Whether to output the image in channel-first
            format (C, H, W). If False, outputs in (H, W, C) format.
            Defaults to True.
        normalize (bool): Whether to normalize the image values by dividing
            by 10000. This is common for Sentinel-2 imagery. Defaults to False.
        select_bands (list[int], optional): List of band indices to select.
            If None, all bands are loaded. Defaults to None.
            Example: [0, 1, 2] to select first 3 bands (BGR for Sentinel-2).

    Examples:
        >>> # Load all bands and convert to float32
        >>> transform = LoadMultispectralImageFromFile(to_float32=True)
        >>> # Load specific bands (e.g., RGB: bands 3,2,1 for Sentinel-2)
        >>> transform = LoadMultispectralImageFromFile(
        ...     to_float32=True,
        ...     select_bands=[3, 2, 1],
        ...     normalize=True
        ... )
    """

    def __init__(
        self,
        to_float32: bool = True,
        channel_first: bool = False,  # Changed default to False for mmpretrain compatibility
        normalize: bool = False,
        select_bands: Optional[list] = None,
    ):
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.normalize = normalize
        self.select_bands = select_bands

    def transform(self, results: dict) -> dict:
        """Transform function to load multispectral image from file.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with loaded image.
        """
        filename = results['img_path']

        # Load image using tifffile
        try:
            img = tifffile.imread(filename)#(12, 120, 120)
        except Exception as e:
            raise RuntimeError(f'Failed to load image from {filename}: {e}')

        # Select specific bands if specified
        if self.select_bands is not None:
            if img.ndim == 3:  # (12, 120, 120)
                # Assume channel-first format from TIFF
                if img.shape[0] < img.shape[-1]:  # likely (C, H, W)
                    img = img[self.select_bands, :, :]
                else:  # likely (H, W, C)
                    img = img[:, :, self.select_bands]
            else:
                raise ValueError(
                    f'Expected 3D image for band selection, got shape {img.shape}'
                )

        # Convert to float32 if needed
        if self.to_float32:
            img = img.astype(np.float32)

        # Normalize if specified (common for Sentinel-2: divide by 10000)
        if self.normalize:
            img = img / 10000.0

        # Convert to channel-first format if needed
        if img.ndim == 3: # (12, 120, 120)
            # Check current format and convert if necessary
            if img.shape[0] > img.shape[-1]:  # likely (H, W, C)
                if self.channel_first:
                    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            else:  # likely (C, H, W)
                if not self.channel_first:
                    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        elif img.ndim == 2:
            # Single channel image
            if self.channel_first:
                img = img[np.newaxis, :, :]  # (H, W) -> (1, H, W)
            else:
                img = img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        results['img'] = img
        results['img_shape'] = img.shape[-2:]  # (H, W)
        results['ori_shape'] = img.shape[-2:]  # (H, W)

        return results

    def __repr__(self) -> str:
        repr_str = (
            f'{self.__class__.__name__}('
            f'to_float32={self.to_float32}, '
            f'channel_first={self.channel_first}, '
            f'normalize={self.normalize}, '
            f'select_bands={self.select_bands})'
        )
        return repr_str


@TRANSFORMS.register_module()
class SelectMultispectralBands(BaseTransform):
    """Select specific bands from a multispectral image.

    This transform selects a subset of bands from the loaded image.
    Useful when you want to use only specific spectral bands.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        band_indices (list[int]): List of band indices to select.
            Example: [3, 2, 1] for RGB in Sentinel-2 (0-indexed).

    Examples:
        >>> # Select RGB bands (indices 3,2,1) from Sentinel-2
        >>> transform = SelectMultispectralBands(band_indices=[3, 2, 1])
    """

    def __init__(self, band_indices: list):
        self.band_indices = band_indices

    def transform(self, results: dict) -> dict:
        """Transform function to select bands.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with selected bands.
        """
        img = results['img']

        if img.ndim == 3:
            # Assume channel-first format (C, H, W)
            if img.shape[0] < img.shape[-1]:  # (C, H, W)
                results['img'] = img[self.band_indices, :, :]
            else:  # (H, W, C)
                results['img'] = img[:, :, self.band_indices]
        else:
            raise ValueError(
                f'Expected 3D image for band selection, got shape {img.shape}'
            )

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(band_indices={self.band_indices})'


@TRANSFORMS.register_module()
class MultiSpectralNormalize(BaseTransform):
    """Normalize multispectral image using various methods.

    This transform normalizes multispectral imagery using different methods:
    - Percentile-based normalization (2nd and 98th percentiles)
    - Statistical normalization using mean ± 2*std (SatMAE/SeCo approach)
    - Standard mean/std normalization

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        method (str): Normalization method. Options:
            - 'percentile': percentile-based normalization (default)
            - 'stat': per-channel normalization using mean ± 2*std
            - 'mean_std': standard mean/std normalization
            Defaults to 'percentile'.
        bands (str): Band configuration to use for percentile method.
            Options: 's2', 'rgb'. Defaults to 's2'.
        mean (list[float], optional): Per-band mean values for mean_std method.
            Defaults to None.
        std (list[float], optional): Per-band std values for mean_std method.
            Defaults to None.
        use_8_bit (bool): Whether to convert to 8-bit [0, 255] range when using
            'stat' method. If False, normalizes to [0, 1]. Defaults to False.

    Examples:
        >>> # Use percentile normalization (default)
        >>> transform = MultiSpectralNormalize(method='percentile', bands='s2')
        >>> # Use statistical normalization (SatMAE/SeCo approach)
        >>> transform = MultiSpectralNormalize(method='stat', use_8_bit=False)
        >>> # Use standard mean/std normalization
        >>> transform = MultiSpectralNormalize(
        ...     method='mean_std', mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]
        ... )
    """

    # Sentinel-2 band statistics (2nd and 98th percentiles)
    # Computed from BigEarthNet-S2 dataset
    # S2_PERCENTILES = {
    #     'min': [80, 120, 110, 130, 200, 300, 400, 500, 550, 300, 100, 80],
    #     'max': [6808, 6000, 5500, 5200, 4800, 4500, 4300, 4200, 4100, 3500, 2500, 1800],
    # }
    # direct from smarties
    S2_PERCENTILES = {
        'min': [1.0, 27.01892781481322, 45.058626301104624, 12.495476066364853, 5.990619300620652, 1.0, 1.0793969725828485, 1.0, 1.0, 1.0, 2.3175878903313945, 4.277889404039971],
        'max': [2811.4529362270814, 2604.7705241174135, 2580.36014000968, 2829.486095366986, 3138.512727440568, 4338.892460929282, 5182.992628573339, 5529.262813718612, 5433.892460929279, 5170.23299517986, 4418.1529202629945, 3387.4430439994744],
    }

    def __init__(
        self,
        method: str = 'percentile',
        bands: str = 's2',
        mean: Optional[list] = None,
        std: Optional[list] = None,
        use_8_bit: bool = False,
    ):
        self.method = method
        self.bands = bands
        self.mean = mean
        self.std = std
        self.use_8_bit = use_8_bit

        # Setup for percentile-based normalization
        if self.method == 'percentile':
            if bands == 's2':
                self.percentile_min = np.array(self.S2_PERCENTILES['min'], dtype=np.float32)
                self.percentile_max = np.array(self.S2_PERCENTILES['max'], dtype=np.float32)
            else:
                # For RGB or other bands, use simple min-max
                self.percentile_min = None
                self.percentile_max = None
        else:
            self.percentile_min = None
            self.percentile_max = None

    def transform(self, results: dict) -> dict:
        """Transform function to normalize image.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with normalized image.
        """
        img = results['img']

        if self.method == 'stat':
            # Statistical normalization using mean ± 2*std (SatMAE/SeCo approach)
            img = self._normalize_stat(img)
        elif self.method == 'mean_std':
            # Standard mean/std normalization
            if self.mean is not None and self.std is not None:
                mean = np.array(self.mean, dtype=np.float32)
                std = np.array(self.std, dtype=np.float32)

                # Apply normalization based on image format
                if img.ndim == 3:
                    if img.shape[0] < img.shape[-1]:  # (C, H, W)
                        mean = mean[:, None, None]
                        std = std[:, None, None]
                    else:  # (H, W, C)
                        mean = mean[None, None, :]
                        std = std[None, None, :]

                img = (img - mean) / std
        elif self.method == 'percentile':
            # Percentile-based normalization
            if self.percentile_min is not None and self.percentile_max is not None:
                # Normalize each band to [0, 1] using percentiles
                if img.ndim == 3:
                    if img.shape[0] < img.shape[-1]:  # (C, H, W)
                        pmin = self.percentile_min[:, None, None]
                        pmax = self.percentile_max[:, None, None]
                    else:  # (H, W, C)
                        pmin = self.percentile_min[None, None, :]
                        pmax = self.percentile_max[None, None, :]

                    # Clip and normalize
                    img = np.clip(img, pmin, pmax)
                    img = (img - pmin) / (pmax - pmin + 1e-8)
        else:
            raise ValueError(
                f"Unknown normalization method: {self.method}. "
                f"Expected 'stat', 'mean_std', or 'percentile'."
            )

        results['img'] = img
        return results

    def _normalize_stat(self, img: np.ndarray) -> np.ndarray:
        """Per-channel statistical normalization using mean ± 2*std.

        This method is adapted from SatMAE and SeCo approaches.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            np.ndarray: Normalized image.
        """
        # Ensure float type for computation
        img = img.astype(np.float32)

        # Determine if image is in channel-first or channel-last format
        if img.ndim == 3:
            if img.shape[0] < img.shape[-1]:  # likely (C, H, W)
                channel_axis = 0
                num_channels = img.shape[0]
            else:  # likely (H, W, C)
                channel_axis = 2
                num_channels = img.shape[-1]
        else:
            raise ValueError(f'Expected 2D or 3D image, got shape {img.shape}')

        # Process each channel separately
        normalized_channels = []
        for i in range(num_channels):
            if channel_axis == 0:
                channel = img[i, :, :]
            else:
                channel = img[:, :, i]

            # Calculate min and max values based on mean ± 2*std
            min_value = channel.mean() - 2 * channel.std()
            max_value = channel.mean() + 2 * channel.std()

            # Normalize
            normalized = (channel - min_value) / (max_value - min_value + 1e-8)

            if self.use_8_bit:
                normalized = normalized * 255.0
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                normalized = np.clip(normalized, 0, 1)

            normalized_channels.append(normalized)

        # Stack channels back together
        if channel_axis == 0:
            img = np.stack(normalized_channels, axis=0)
        else:
            img = np.stack(normalized_channels, axis=2)

        return img

    def __repr__(self) -> str:
        repr_str = (
            f'{self.__class__.__name__}('
            f'method={self.method}, '
            f'bands={self.bands}, '
            f'mean={self.mean}, '
            f'std={self.std}, '
            f'use_8_bit={self.use_8_bit})'
        )
        return repr_str


@TRANSFORMS.register_module()
class MultiSpectralResize(BaseTransform):
    """Resize multispectral images.

    This transform resizes multi-channel images using various interpolation methods.
    Unlike mmcv's Resize which is optimized for RGB (3 channels), this supports
    arbitrary number of channels (e.g., 12-channel Sentinel-2 imagery).

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        scale (int or tuple): Target size. If int, resize to (scale, scale).
            If tuple, resize to (height, width).
        interpolation (str): Interpolation method. Options:
            - 'nearest': Nearest neighbor
            - 'bilinear': Bilinear interpolation (default)
            - 'bicubic': Bicubic interpolation (higher quality, slower)
            - 'area': Resampling using pixel area relation
        keep_ratio (bool): Whether to keep aspect ratio. Defaults to False.

    Examples:
        >>> # Resize to 224x224 using bilinear interpolation
        >>> transform = MultiSpectralResize(scale=224)
        >>> # Resize to 256x128 using bicubic interpolation
        >>> transform = MultiSpectralResize(scale=(256, 128), interpolation='bicubic')
    """

    _INTERPOLATION_MAPPING = {
        'nearest': 0,   # cv2.INTER_NEAREST
        'bilinear': 1,  # cv2.INTER_LINEAR
        'bicubic': 2,   # cv2.INTER_CUBIC
        'area': 3,      # cv2.INTER_AREA
    }

    def __init__(
        self,
        scale,
        interpolation: str = 'bilinear',
        keep_ratio: bool = False,
    ):
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = scale

        self.interpolation = interpolation
        self.keep_ratio = keep_ratio

        # Validate interpolation method
        if interpolation not in self._INTERPOLATION_MAPPING:
            raise ValueError(
                f'Invalid interpolation method: {interpolation}. '
                f'Must be one of {list(self._INTERPOLATION_MAPPING.keys())}'
            )

    def transform(self, results: dict) -> dict:
        """Transform function to resize image.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with resized image.
        """
        import cv2

        img = results['img']

        # Get target size
        target_h, target_w = self.scale

        if self.keep_ratio:
            # Calculate scale to keep aspect ratio
            h, w = img.shape[:2] if img.ndim == 3 else img.shape
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = target_h, target_w

        # Get interpolation flag
        interp_flag = self._INTERPOLATION_MAPPING[self.interpolation]

        # Resize based on image format
        if img.ndim == 2:
            # Single channel: (H, W)
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)
        elif img.ndim == 3:
            # Multi-channel: could be (H, W, C) or (C, H, W)
            if img.shape[0] < img.shape[-1]:  # likely (C, H, W)
                # Transpose to (H, W, C) for cv2.resize
                img_hwc = np.transpose(img, (1, 2, 0))
                resized = cv2.resize(img_hwc, (new_w, new_h), interpolation=interp_flag)
                # Transpose back to (C, H, W)
                resized = np.transpose(resized, (2, 0, 1))
            else:  # likely (H, W, C)
                resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)

                # cv2.resize may squeeze single channel dimension
                if resized.ndim == 2 and img.ndim == 3:
                    resized = resized[:, :, np.newaxis]
        else:
            raise ValueError(f'Unsupported image dimensions: {img.ndim}')

        # Ensure float32 dtype is preserved
        if img.dtype == np.float32:
            resized = resized.astype(np.float32)

        results['img'] = resized
        results['img_shape'] = resized.shape[:2]  # (H, W)

        return results

    def __repr__(self) -> str:
        repr_str = (
            f'{self.__class__.__name__}('
            f'scale={self.scale}, '
            f'interpolation={self.interpolation}, '
            f'keep_ratio={self.keep_ratio})'
        )
        return repr_str

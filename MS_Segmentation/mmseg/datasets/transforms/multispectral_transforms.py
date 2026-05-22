"""
Custom transforms for multispectral data processing
"""
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


@TRANSFORMS.register_module()
class LoadMultispectralImageFromFile(BaseTransform):
    """Load multispectral images from file.

    Required keys:
        - img_path

    Modified keys:
        - img
        - img_shape
        - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
        color_type (str): Not used for multispectral, kept for compatibility.
        imdecode_backend (str): Backend for image decoding.
    """

    def __init__(self,
                 to_float32=True,
                 color_type='unchanged',
                 imdecode_backend='tifffile'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def transform(self, results):
        """Functions to load multispectral image.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']

        if self.imdecode_backend == 'tifffile':
            import tifffile
            img = tifffile.imread(filename)  # Shape: (bands, height, width)
            img = np.transpose(img, (1, 2, 0))  # to (height, width, bands)
        else:
            raise ValueError(f'Unsupported backend: {self.imdecode_backend}')

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}')")
        return repr_str


@TRANSFORMS.register_module()
class LoadMultispectralAnnotations(BaseTransform):
    """Load semantic segmentation annotations from .tif files.

    Required keys:
        - seg_map_path

    Modified keys:
        - gt_seg_map

    Args:
        reduce_zero_label (bool): Whether to mark label zero as ignored.
        imdecode_backend (str): Backend for image decoding.
    """

    def __init__(self,
                 reduce_zero_label=False,
                 imdecode_backend='tifffile'):
        self.reduce_zero_label = reduce_zero_label
        self.imdecode_backend = imdecode_backend

    def transform(self, results):
        """Load segmentation map from file.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation map.
        """
        filename = results['seg_map_path']

        if self.imdecode_backend == 'tifffile':
            import tifffile
            gt_semantic_seg = tifffile.imread(filename)
        else:
            raise ValueError(f'Unsupported backend: {self.imdecode_backend}')

        # Reduce zero label if needed
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results['gt_seg_map'] = gt_semantic_seg
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'reduce_zero_label={self.reduce_zero_label}, '
                    f"imdecode_backend='{self.imdecode_backend}')")
        return repr_str


@TRANSFORMS.register_module()
class MultispectralNormalize(BaseTransform):
    """Normalize multispectral images with mean and std.

    Required keys:
        - img

    Modified keys:
        - img

    Args:
        mean (sequence): Mean values of different bands.
        std (sequence): Std values of different bands.
        to_rgb (bool): Not used for multispectral, kept for compatibility.
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results.
        """
        img = results['img']
        assert img.shape[2] == len(self.mean), \
            f'Image has {img.shape[2]} bands, but mean has {len(self.mean)} values'

        results['img'] = (img - self.mean) / self.std
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'mean={self.mean.tolist()}, '
        repr_str += f'std={self.std.tolist()}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class SelectBands(BaseTransform):
    """Select specific bands from multispectral image.

    Required keys:
        - img

    Modified keys:
        - img

    Args:
        band_indices (list): List of band indices to select.
    """

    def __init__(self, band_indices):
        self.band_indices = band_indices

    def transform(self, results):
        """Call function to select bands.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with selected bands.
        """
        img = results['img']
        results['img'] = img[:, :, self.band_indices]
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'band_indices={self.band_indices})'
        return repr_str


@TRANSFORMS.register_module()
class AddProjIndices(BaseTransform):
    """Add projection indices for SMARTIES model.

    The projection indices map each band to its spectral projection.
    Based on electromagnetic_spectrum.yaml configuration.

    Required keys:
        - img

    Added keys:
        - proj_indices

    Args:
        proj_indices (list): List of projection indices for each band.
    """

    def __init__(self, proj_indices):
        self.proj_indices = proj_indices

    def transform(self, results):
        """Call function to add projection indices.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with projection indices added.
        """
        # Projection indices are the same for all pixels in a batch
        # We'll add it as metadata to be batched later
        results['proj_indices'] = np.array(self.proj_indices, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'proj_indices={self.proj_indices})'
        return repr_str


@TRANSFORMS.register_module()
class PercentileClip(BaseTransform):
    """Clip multispectral image values based on percentiles.

    Required keys:
        - img

    Modified keys:
        - img

    Args:
        min_percentile (float or list): Lower percentile value(s) for clipping.
        max_percentile (float or list): Upper percentile value(s) for clipping.
        per_band (bool): If True, clip each band independently. Default: True.
    """

    def __init__(self, min_percentile=1.0, max_percentile=99.0, per_band=True):
        if isinstance(min_percentile, (int, float)):
            self.min_percentile = [min_percentile]
        else:
            self.min_percentile = min_percentile

        if isinstance(max_percentile, (int, float)):
            self.max_percentile = [max_percentile]
        else:
            self.max_percentile = max_percentile

        self.per_band = per_band

    def transform(self, results):
        """Call function to clip image values.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Clipped results.
        """
        img = results['img']

        if self.per_band:
            # Clip each band independently
            for i in range(img.shape[2]):
                min_val = np.percentile(img[:, :, i], self.min_percentile[0] if len(self.min_percentile) == 1 else self.min_percentile[i])
                max_val = np.percentile(img[:, :, i], self.max_percentile[0] if len(self.max_percentile) == 1 else self.max_percentile[i])
                img[:, :, i] = np.clip(img[:, :, i], min_val, max_val)
        else:
            # Clip all bands using global percentiles
            min_val = np.percentile(img, self.min_percentile[0])
            max_val = np.percentile(img, self.max_percentile[0])
            img = np.clip(img, min_val, max_val)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'min_percentile={self.min_percentile}, '
        repr_str += f'max_percentile={self.max_percentile}, '
        repr_str += f'per_band={self.per_band})'
        return repr_str


@TRANSFORMS.register_module()
class PackMultispectralSegInputs(BaseTransform):
    """Pack the inputs data for multispectral segmentation.

    Required keys:
        - img
        - gt_seg_map
        - proj_indices

    Added keys:
        - inputs (dict): The forward data of models, containing:
            - imgs: Image tensor
            - proj_indices: Projection indices tensor

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed.
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results):
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Packed data and meta information.
        """
        packed_results = dict()

        # Pack image
        img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        # HWC to CHW
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img_tensor = torch.from_numpy(img)
        packed_results['inputs'] = img_tensor

        # Pack projection indices as a separate key for data preprocessor
        if 'proj_indices' in results:
            proj_indices_tensor = torch.from_numpy(results['proj_indices'])
            packed_results['proj_indices'] = proj_indices_tensor

        # Create SegDataSample
        data_sample = SegDataSample()

        # Pack gt_sem_seg
        if 'gt_seg_map' in results:
            gt_semantic_seg = results['gt_seg_map']
            gt_sem_seg_data = dict(
                data=torch.from_numpy(gt_semantic_seg).long())
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # Pack meta info
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

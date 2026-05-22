"""Transforms for SMARTIES model to add projection indices to data samples."""
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadProjectionIndices(BaseTransform):
    """Load projection indices for SMARTIES model.

    This transform adds projection indices (proj_indices) to the data sample's
    metainfo. The projection indices map each spectral band to its corresponding
    electromagnetic spectrum projection index in the SMARTIES model.

    Args:
        dataset_name (str): Name of the dataset to automatically load
            projection indices. Supported datasets: 'DFC2020', 'BigEarthNetS2',
            'EuroSAT', etc. Defaults to 'DFC2020'.
        proj_indices (list or np.ndarray, optional): Manual projection indices
            to use instead of loading from dataset config. If provided,
            this overrides dataset_name. Defaults to None.

    Example:
        >>> # In your data pipeline config
        >>> train_pipeline = [
        ...     dict(type='LoadImageFromFile'),
        ...     dict(type='LoadAnnotations'),
        ...     dict(type='LoadProjectionIndices', dataset_name='DFC2020'),
        ...     dict(type='PackSegInputs')
        ... ]
    """

    def __init__(self,
                 dataset_name: str = 'DFC2020',
                 proj_indices: list = None):
        """Initialize LoadProjectionIndices transform.

        Args:
            dataset_name (str): Name of the dataset.
            proj_indices (list or np.ndarray, optional): Manual projection indices.
        """
        self.dataset_name = dataset_name
        self._manual_proj_indices = proj_indices

        # Load projection indices if not manually specified
        if proj_indices is None:
            try:
                from mmseg.models.utils.smarties_utils import get_proj_indices_for_dataset
                self.proj_indices = get_proj_indices_for_dataset(dataset_name)
                print(f'Loaded projection indices for {dataset_name}: {self.proj_indices}')
            except Exception as e:
                raise ValueError(
                    f'Could not load projection indices for dataset {dataset_name}. '
                    f'Error: {e}. Please provide proj_indices manually.') from e
        else:
            self.proj_indices = np.array(proj_indices)
            print(f'Using manual projection indices: {self.proj_indices}')

    def transform(self, results: dict) -> dict:
        """Transform function to add proj_indices to results.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict with proj_indices in metainfo.
        """
        # Add proj_indices to results for later use
        results['proj_indices'] = self.proj_indices.copy()

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(dataset_name={self.dataset_name}, '
        repr_str += f'proj_indices={self.proj_indices})'
        return repr_str


@TRANSFORMS.register_module()
class AddProjectionIndicesToMetainfo(BaseTransform):
    """Add projection indices to the metainfo of packed data sample.

    This transform should be used after PackSegInputs to ensure
    proj_indices are included in the SegDataSample's metainfo.

    Note: This is typically not needed if you use the PackSegInputs
    transform properly, as it should automatically include proj_indices
    from the results dict into metainfo.

    Args:
        keys (list[str]): Keys to add to metainfo. Defaults to ['proj_indices'].

    Example:
        >>> train_pipeline = [
        ...     dict(type='LoadImageFromFile'),
        ...     dict(type='LoadAnnotations'),
        ...     dict(type='LoadProjectionIndices', dataset_name='DFC2020'),
        ...     dict(type='PackSegInputs', meta_keys=['proj_indices']),
        ... ]
    """

    def __init__(self, keys: list = None):
        """Initialize the transform.

        Args:
            keys (list[str]): Keys to add to metainfo.
        """
        self.keys = keys if keys is not None else ['proj_indices']

    def transform(self, results: dict) -> dict:
        """Add specified keys to metainfo.

        Args:
            results (dict): Result dict containing packed data sample.

        Returns:
            dict: Updated result dict.
        """
        # Check if data_samples exists (after PackSegInputs)
        if 'data_samples' in results:
            data_sample = results['data_samples']
            # Add each key from results to metainfo
            for key in self.keys:
                if key in results:
                    data_sample.set_metainfo({key: results[key]})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        return repr_str

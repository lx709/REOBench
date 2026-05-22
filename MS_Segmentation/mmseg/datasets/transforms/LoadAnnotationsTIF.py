from mmseg.registry import TRANSFORMS
import tifffile
import numpy as np

@TRANSFORMS.register_module()
class LoadAnnotationsTIF:

    def __call__(self, results):
        path = results['seg_map_path']
        arr = tifffile.imread(path)

        # MMSeg 要求 labels 是 int64
        results['gt_seg_map'] = arr.astype(np.int64)

        results['seg_fields'] = ['gt_seg_map']

        return results

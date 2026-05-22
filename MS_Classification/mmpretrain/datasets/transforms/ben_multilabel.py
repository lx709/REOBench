from mmengine.registry import TRANSFORMS
import numpy as np
import torch

@TRANSFORMS.register_module()
class BENMultiLabelTransform:

    def __call__(self, results):
        label = np.asarray(results['gt_multilabel'], dtype=np.float32)
        results['gt_multilabel'] = torch.from_numpy(label)
        return results

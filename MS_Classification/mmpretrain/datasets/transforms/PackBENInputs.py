
from mmpretrain.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
@TRANSFORMS.register_module()
class PackBENInputs(BaseTransform):

    def transform(self, results):
        img = results['img']  # (C, H, W)
        
        packed = {
            'inputs': img,
            'data_samples': {
                'img_shape': results['img_shape'],
                'ori_shape': results['ori_shape'],
                'img_path': results['img_path'],
                'gt_label': results.get('gt_label', None),
            }
        }
        return packed

# load_ben_image.py
import os
import tifffile
import numpy as np
from mmpretrain.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadBENImage(BaseTransform):

    def __init__(self, image_root, max_value=10000, to_float32=True):
        self.image_root = image_root
        self.max_value = max_value
        self.to_float32 = to_float32

    def transform(self, results):
        rel = results['img_path']
        path = os.path.join(self.image_root, rel)

        img = tifffile.imread(path)  # (12, H, W)
        # print("======== PER BAND STAT ========")
        # for i in range(img.shape[0]):
        #     print(f"band {i}: max={img[i].max()}, min={img[i].min()}, mean={img[i].mean():.2f}")
        # print("===============================")
        # import pdb;pdb.set_trace()
        if self.to_float32:
            img = img.astype(np.float32)
        # print("Loaded BEN shape:", img.shape)

        
        mins = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        maxs = np.asarray([9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ], dtype=np.float32)  
        # img = img / self.max_value
        # mean=np.asarray([0.11504939943552017, 0.1482059806585312, 0.21661478281021118, 0.198346346616745, 0.2958453893661499, 0.4101085066795349, 0.3985908329486847, 0.399421751499176, 0.41560056805610657, 0.43314507603645325, 0.3597027063369751, 0.29588255286216736])
        # std=np.asarray([0.14282256364822388, 0.14939884841442108, 0.16681823134422302, 0.1924632489681244, 0.19392383098602295, 0.23844216763973236, 0.23693138360977173, 0.24099795520305634, 0.2424980103969574, 0.2453242689371109, 0.2402467578649521, 0.23555098474025726])
        img = (img - mins[:, None, None]) / (maxs[:,None,None] - mins[:,None,None])
        # print(img.min(), img.max(), img.mean(), img.std())

        # 必须补充这两个，否则 Resize / PackInputs 会误判维度！！
        results['ori_shape'] = img.shape       # (12, H, W)
        results['img_shape'] = img.shape       # (12, H, W)

        results['img'] = img
        results['img_path'] = path

        # append 而不是覆盖
        results.setdefault('img_fields', []).append('img')

        return results

import numpy as np
import tifffile
from mmseg.registry import TRANSFORMS
import torch


# @TRANSFORMS.register_module()
# class LoadMultiChannelTIF:
#     def __init__(self,
#                  bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                  to_float32=True):
#         self.bands = bands
#         self.to_float32 = to_float32

#     def __call__(self, results):
#         filename = results['img_info']['filename']
#         # print(filename)
#         # import pdb;pdb.set_trace()

#         img = tifffile.imread(filename)

#         img = np.transpose(img, (1,2,0))

#         if self.bands is not None:
#             img = img[:, :, self.bands]

#         if self.to_float32:
#             img = img.astype(np.float32)
#         img = img / 255.0
#         # print("Loaded img shape:", img.shape)
#         # img = torch.from_numpy(img)

#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         return results
@TRANSFORMS.register_module()
class LoadMultiChannelTIF:

    def __init__(self,
                 bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 to_float32=True,
                 norm=True):
        self.bands = bands
        self.to_float32 = to_float32
        self.norm = norm

    def __call__(self, results):

        filename = results['img_info']['filename']

        img = tifffile.imread(results['img_info']['filename'])   # (C,H,W)
        # print("======== PER BAND STAT ========")
        # for i in range(img.shape[0]):
        #     print(f"band {i}: max={img[i].max()}, min={img[i].min()}, mean={img[i].mean():.2f}")
        # print("===============================")
        # import pdb;pdb.set_trace()
        # print(img.shape)

        # 选通道
        if self.bands is not None:
            img = img[self.bands]                             # (C,H,W)

        # 转 float
        if self.to_float32:
            img = img.astype(np.float32)

        # normalize
        if self.norm:
            img = img / 255.0

        c, h, w = img.shape

        results['img'] = img
        results['img_path'] = filename
        # results['img_shape'] = img.shape
        # # results['ori_shape'] = img.shape
        # results['ori_shape'] = (img.shape[1],img.shape[2])
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        results['pad_shape'] = (h, w)
        results['ori_shape'] = tuple(results['ori_shape'])
        results['img_shape'] = tuple(results['img_shape'])
        results['pad_shape'] = tuple(results['pad_shape'])
        return results

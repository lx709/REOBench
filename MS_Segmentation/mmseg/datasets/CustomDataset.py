from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os
import tifffile
import numpy as np

# @DATASETS.register_module()
# class CustomDataset(BaseSegDataset):

#     def __init__(self, img_dir, ann_dir, pipeline, data_root=None, **kwargs):
#         self.img_dir = img_dir
#         self.ann_dir = ann_dir
#         super().__init__(data_root=data_root, pipeline=pipeline, **kwargs)

#     def load_data_list(self):
#         img_dir = os.path.join(self.data_root, self.img_dir)
#         ann_dir = os.path.join(self.data_root, self.ann_dir)

#         imgs = sorted(os.listdir(img_dir))

#         data_list = []

#         for fn in imgs:
#             img_path = os.path.join(img_dir, fn)
#             seg_path = os.path.join(ann_dir, fn)

#             data_list.append(dict(
#                 img=img_path,
#                 seg_map=seg_path,
#                 img_info=dict(filename=img_path)
#             ))

#         return data_list

@DATASETS.register_module()
class CustomDataset(BaseSegDataset):

    def __init__(self,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 pipeline=None,
                 **kwargs):

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        

        super().__init__(data_root=data_root,
                         pipeline=pipeline,
                         **kwargs)
        # self.dataset_meta = dict(
        #     classes = tuple(str(i) for i in range(8))
        # )

    def load_data_list(self):

        img_dir = os.path.join(self.data_root, self.img_dir)
        ann_dir = os.path.join(self.data_root, self.ann_dir)

        imgs = sorted(os.listdir(img_dir))

        data_list = []

        for fn in imgs:
            img = os.path.join(img_dir, fn)
            seg = os.path.join(ann_dir, fn)

            data_list.append(dict(
                img = img,
                seg_map = seg,
                seg_map_path = seg,
                img_info = dict(filename=img)
            ))

        return data_list

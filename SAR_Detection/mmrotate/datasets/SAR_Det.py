# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
# import os.path as osp
# from typing import List, Union

# from mmengine.dataset import BaseDataset
# from mmengine.fileio import get_local_path
# from mmdet.registry import DATASETS as MMDATASETS

# from mmdet.datasets.api_wrappers import COCO
# from mmrotate.registry import DATASETS as MMROTATE_DATASETS


# @MMROTATE_DATASETS.register_module()
# @MMDATASETS.register_module()
# class SAR_Det_Finegrained_Dataset(BaseDataset):
#     """SARDet 100k 细粒度检测数据集。"""

#     METAINFO = {
#         'classes': ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
#         'palette': [
#             (220, 20, 60), (0, 0, 230), (106, 0, 228),
#             (0, 182, 0), (200, 182, 0), (0, 182, 200)
#         ]
#     }
#     COCOAPI = COCO
#     ANN_ID_UNIQUE = True

#     def load_data_list(self) -> List[dict]:
#         """Load annotations from a COCO-style annotation file."""
#         # with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
#         with get_local_path(self.ann_file,) as local_path:
#             self.coco = self.COCOAPI(local_path)
#         self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
#         self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
#         self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

#         img_ids = self.coco.get_img_ids()
#         data_list = []
#         total_ann_ids = []
#         for img_id in img_ids:
#             raw_img_info = self.coco.load_imgs([img_id])[0]
#             raw_img_info['img_id'] = img_id

#             ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#             raw_ann_info = self.coco.load_anns(ann_ids)
#             total_ann_ids.extend(ann_ids)

#             parsed_data_info = self.parse_data_info({
#                 'raw_ann_info': raw_ann_info,
#                 'raw_img_info': raw_img_info
#             })
#             data_list.append(parsed_data_info)

#         if self.ANN_ID_UNIQUE:
#             assert len(set(total_ann_ids)) == len(total_ann_ids), \
#                 f"Annotation ids in '{self.ann_file}' are not unique!"

#         del self.coco
#         return data_list

#     def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
#         """Parse raw annotation to target format."""
#         img_info = raw_data_info['raw_img_info']
#         ann_info = raw_data_info['raw_ann_info']

#         data_info = {}
#         img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
#         data_info['img_path'] = img_path
#         data_info['img_id'] = img_info['img_id']
#         data_info['seg_map_path'] = None
#         data_info['height'] = img_info['height']
#         data_info['width'] = img_info['width']

#         if getattr(self, 'return_classes', False):
#             data_info['text'] = self.metainfo['classes']
#             data_info['custom_entities'] = True

#         instances = []
#         for ann in ann_info:
#             if ann.get('ignore', False):
#                 continue
#             x1, y1, w, h = ann['bbox']
#             inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
#             inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
#             if inter_w * inter_h == 0:
#                 continue
#             if ann['area'] <= 0 or w < 1 or h < 1:
#                 continue
#             if ann['category_id'] not in self.cat_ids:
#                 continue

#             bbox = [x1, y1, x1 + w, y1 + h]
#             ignore_flag = 1 if ann.get('iscrowd', False) else 0
#             instance = {
#                 'bbox': bbox,
#                 'bbox_label': self.cat2label[ann['category_id']],
#                 'ignore_flag': ignore_flag
#             }
#             if ann.get('segmentation', None):
#                 instance['mask'] = ann['segmentation']
#             instances.append(instance)

#         data_info['instances'] = instances
#         return data_info

#     def filter_data(self) -> List[dict]:
#         """Filter annotations according to filter_cfg."""
#         if self.test_mode:
#             return self.data_list

#         if self.filter_cfg is None:
#             return self.data_list

#         filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
#         min_size = self.filter_cfg.get('min_size', 0)

#         ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
#         ids_in_cat = set()
#         for class_id in self.cat_ids:
#             ids_in_cat |= set(self.cat_img_map[class_id])
#         ids_in_cat &= ids_with_ann

#         valid_data_infos = []
#         for data_info in self.data_list:
#             width = data_info['width']
#             height = data_info['height']
#             if filter_empty_gt and data_info['img_id'] not in ids_in_cat:
#                 continue
#             if min(width, height) >= min_size:
#                 valid_data_infos.append(data_info)

#         return valid_data_infos

#     def get_cat_ids(self, idx: int) -> List[int]:
#         instances = self.get_data_info(idx)['instances']
#         return [instance['bbox_label'] for instance in instances]
import copy
import os.path as osp
from typing import List, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS as MMDATASETS
from mmdet.datasets.api_wrappers import COCO
from mmrotate.registry import DATASETS as MMROTATE_DATASETS


@MMROTATE_DATASETS.register_module()
@MMDATASETS.register_module()
class SAR_Det_Finegrained_Dataset(BaseDataset):
    """SARDet 100k 细粒度检测数据集（直接使用 polygon bbox 四点版本）。"""

    METAINFO = {
        'classes': ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
        'palette': [
            (220, 20, 60), (0, 0, 230), (106, 0, 228),
            (0, 182, 0), (200, 182, 0), (0, 182, 200)
        ]
    }
    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from a COCO-style annotation file."""
        with get_local_path(self.ann_file) as local_path:
            self.coco = self.COCOAPI(local_path)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), \
                f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format. Assumes ann['bbox'] is polygon four points."""
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if getattr(self, 'return_classes', False):
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            if ann.get('ignore', False):
                continue

            # -----------------------
            # 直接使用 polygon 四点 bbox
            bbox_polygon = np.array(ann['bbox'], dtype=np.float32)

            # 可选：确保 polygon 在图像内
            if np.any(bbox_polygon[:, 0] < 0) or np.any(bbox_polygon[:, 0] > img_info['width']):
                continue
            if np.any(bbox_polygon[:, 1] < 0) or np.any(bbox_polygon[:, 1] > img_info['height']):
                continue
            # -----------------------

            ignore_flag = 1 if ann.get('iscrowd', False) else 0
            instance = {
                'bbox': bbox_polygon,        # polygon 四点
                'bbox_label': self.cat2label[ann['category_id']],
                'ignore_flag': ignore_flag
            }
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg."""
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for data_info in self.data_list:
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and data_info['img_id'] not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
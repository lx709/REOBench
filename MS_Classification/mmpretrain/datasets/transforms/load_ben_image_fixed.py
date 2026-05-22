# load_ben_image_fixed.py
# 修复版本：使用 CROMA 官方的归一化方式（来自 SatMAE 和 SeCo）

import os
import tifffile
import numpy as np
from mmpretrain.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class LoadBENImage(BaseTransform):
    """
    加载 BigEarthNet 多光谱图像，并使用 CROMA 官方的归一化方式。
    
    CROMA 归一化策略（来自 SatMAE/SeCo）：
    - 对每个通道独立计算 mean 和 std
    - min_value = mean - 2 * std
    - max_value = mean + 2 * std
    - 归一化到 [0, 1] 并 clip
    
    Args:
        image_root: 图像根目录
        per_sample_norm: 是否按每个样本独立归一化（推荐 True，与 CROMA 预训练一致）
        use_8_bit: 是否转换为 8-bit（0-255），然后再除以 255
    """

    def __init__(self, image_root, per_sample_norm=True, use_8_bit=False):
        self.image_root = image_root
        self.per_sample_norm = per_sample_norm
        self.use_8_bit = use_8_bit

    def normalize_croma_style(self, img):
        """
        CROMA 官方归一化方式
        img: shape (C, H, W), numpy array
        """
        img = img.astype(np.float32)
        normalized_channels = []
        
        for c in range(img.shape[0]):
            channel = img[c]
            
            if self.per_sample_norm:
                # 按当前样本的当前通道计算（与 CROMA 预训练一致）
                mean_val = channel.mean()
                std_val = channel.std()
            else:
                # 如果你想用全局统计量，需要预先计算好
                # 但这与 CROMA 预训练不一致，不推荐
                mean_val = channel.mean()
                std_val = channel.std()
            
            min_value = mean_val - 2 * std_val
            max_value = mean_val + 2 * std_val
            
            # 避免除零
            if max_value - min_value < 1e-6:
                normalized = np.zeros_like(channel)
            else:
                normalized = (channel - min_value) / (max_value - min_value)
            
            # Clip 到 [0, 1]
            normalized = np.clip(normalized, 0, 1)
            
            if self.use_8_bit:
                # 转换为 8-bit，然后最后再除以 255
                normalized = (normalized * 255).astype(np.uint8).astype(np.float32) / 255.0
            
            normalized_channels.append(normalized)
        
        result = np.stack(normalized_channels, axis=0)
        # 确保内存连续，避免 PyTorch 的 negative stride 错误
        return np.ascontiguousarray(result)

    def transform(self, results):
        rel = results['img_path']
        path = os.path.join(self.image_root, rel)

        img = tifffile.imread(path)  # 通常是 (12, H, W) 或 (H, W, 12)
        
        # 确保是 (C, H, W) 格式
        if img.ndim == 3:
            if img.shape[2] == 12:  # (H, W, C) -> (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                img = np.ascontiguousarray(img)  # 确保内存连续
        
        # 使用 CROMA 风格的归一化
        img = self.normalize_croma_style(img)
        
        # Debug 打印（训练时可以注释掉）
        # print(f"After normalization: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")
        
        results['ori_shape'] = img.shape       # (12, H, W)
        results['img_shape'] = img.shape       # (12, H, W)
        results['img'] = img
        results['img_path'] = path
        results.setdefault('img_fields', []).append('img')

        return results


@TRANSFORMS.register_module()
class EnsureContiguous(BaseTransform):
    """
    确保图像数组是内存连续的。
    放在 RandomFlip 等可能产生 negative stride 的 Transform 之后，PackInputs 之前。
    
    用法:
        train_pipeline = [
            dict(type='LoadBENImage', ...),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='EnsureContiguous'),  # 加在这里
            dict(type='PackInputs')
        ]
    """
    
    def __init__(self, keys=None):
        self.keys = keys or ['img']
    
    def transform(self, results):
        for key in self.keys:
            if key in results:
                img = results[key]
                if isinstance(img, np.ndarray) and not img.flags.c_contiguous:
                    results[key] = np.ascontiguousarray(img)
        return results


@TRANSFORMS.register_module()
class LoadBENImageV2(BaseTransform):
    """
    另一种可选方案：使用全局统计量归一化
    
    如果你的数据集已经有预计算的全局 mean/std，可以用这个版本。
    但注意这与 CROMA 预训练的归一化方式不完全一致。
    """

    def __init__(self, image_root, 
                 mean=None, 
                 std=None,
                 clip_range=(0, 1)):
        self.image_root = image_root
        self.clip_range = clip_range
        
        # BigEarthNet-S2 的默认全局统计量（需要你根据自己的数据集验证）
        # 这些值是示例，你需要从你的数据集中计算
        if mean is None:
            self.mean = np.array([
                340.76769064, 429.9430203, 614.21682446, 590.23569706,
                950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
                2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131
            ], dtype=np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
            
        if std is None:
            self.std = np.array([
                554.81258967, 572.41639287, 582.87945694, 675.88746967,
                729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
                1356.13789355, 1302.3292881, 1079.19066363, 818.86747235
            ], dtype=np.float32)
        else:
            self.std = np.array(std, dtype=np.float32)

    def transform(self, results):
        rel = results['img_path']
        path = os.path.join(self.image_root, rel)

        img = tifffile.imread(path)
        
        if img.ndim == 3 and img.shape[2] == 12:
            img = np.transpose(img, (2, 0, 1))
        
        img = img.astype(np.float32)
        
        # 类似 CROMA 的 mean ± 2*std 归一化
        for c in range(12):
            min_val = self.mean[c] - 2 * self.std[c]
            max_val = self.mean[c] + 2 * self.std[c]
            img[c] = (img[c] - min_val) / (max_val - min_val + 1e-6)
        
        img = np.clip(img, self.clip_range[0], self.clip_range[1])
        
        results['ori_shape'] = img.shape
        results['img_shape'] = img.shape
        results['img'] = img
        results['img_path'] = path
        results.setdefault('img_fields', []).append('img')

        return results
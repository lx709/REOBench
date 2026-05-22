import torch.nn as nn
from mmpretrain.registry import MODELS

@MODELS.register_module()
class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * self.loss_fn(pred, target)

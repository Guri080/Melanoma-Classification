import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, weights = None, alpha:float = 0.25, gamma: float = 2.0, reduction:str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            if self.weights is not None:
                weights_per_sample = self.weights[targets]
                return (focal_loss * weights_per_sample).sum() / weights_per_sample.sum()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




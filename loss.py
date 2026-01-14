import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha:float = 0.25, gamma: float = 2.0, reduction:str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, 
                                targets, 
                                self.alpha, 
                                self.gamma, 
                                self.reduction)




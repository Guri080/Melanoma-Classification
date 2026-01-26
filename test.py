import torch
import torch.nn as nn
import models


model, _ = models.EfficientNet(3, 2)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# Trainable parameters only
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", trainable_params)

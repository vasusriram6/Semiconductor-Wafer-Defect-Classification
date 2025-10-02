import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_ft = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_ft, num_classes)
    return model
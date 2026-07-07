import torch.nn as nn
from torchvision import models

def build_model(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # جایگزینی لایه آخر برای تعداد کلاس‌های ما
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

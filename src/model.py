import torch.nn as nn
from torchvision import models


def get_model(backbone: str, num_classes: int, pretrained: bool = True, finetune: str = "full"):
    """
    backbone : resnet50 | convnext_base
    finetune : full | frozen
    """
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        if finetune == "frozen":
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)

        if finetune == "frozen":
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model

import torchvision.models as visionmodels
import torch.nn as nn
import torch.optim as optim
import torch

class Emolens(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = visionmodels.efficientnet_b0(weights=visionmodels.EfficientNet_B0_Weights.DEFAULT)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 7))

    def forward(self, x):
        x = self.model(x)

        return x
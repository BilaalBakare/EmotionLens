import torchvision.models as visionmodels
import torch.nn as nn
import torch.optim as optim
import torch

class Emolens(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = visionmodels.efficientnet_b4(weights=None)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for parameter in self.model.features[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.model.features[-2].parameters():
            parameter.requires_grad = True
        for parameter in self.model.features[-3].parameters():
            parameter.requires_grad = True
        for parameter in self.model.features[-4].parameters():
            parameter.requires_grad = True
        for parameter in self.model.features[-5].parameters():
            parameter.requires_grad = True
        for parameter in self.model.features[-6].parameters():
            parameter.requires_grad = True


        self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1792, 7))

    def forward(self, x):
        x = self.model(x)

        return x
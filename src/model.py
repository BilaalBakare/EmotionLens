import torchvision.models as visionmodels
import torch.nn as nn
import torch.optim as optim
import torch

class Emolens(nn.module):
    def __init__(self):
        super().__init()

        self.model = visionmodels.efficientnet_b0(weights=visionmodels.EfficientNet_B0_Weights.DEFAULT)

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 7))

    def fit(self, train, num_of_epochs=10):
        criterion = nn.CrossEntropyLoss()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(model_parameters, lr=0.00001)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        for epoch in range(num_of_epochs):
            sum_of_losses = 0
            for images, labels in train:
                images = images.to(device)
                labels = labels.to(device)
                
                y_pred = model(images)
                loss = criterion(y_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_of_losses += loss.item()

            print(f'Epoch {epoch+1}: {sum_of_losses / len(train): .3f}')
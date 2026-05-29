import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import Emolens

train_Data = '/content/train'
Test_Data = '/content/test'

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(root=train_Data, transform=train_transform)
test_dataset = datasets.ImageFolder(root=Test_Data, transform=test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = Emolens()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_parameters, lr=0.00001)

num_of_epochs=10

for epoch in range(num_of_epochs):
    sum_of_train_losses = 0
    sum_of_test_losses = 0
    total_correct = 0
    total_images = 0

    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        y_pred = model(images)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_of_train_losses += loss.item()

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            y_pred = torch.argmax(y_pred, dim=1)
            total_correct += (y_pred == labels).sum().item()
            total_images += labels.size(0)

            sum_of_test_losses += loss.item()

    accuracy = (total_correct / total_images) * 100
    print(f'Epoch {epoch+1}: Train Loss: {sum_of_train_losses / len(train_dataloader): .3f} | Test Loss: {sum_of_test_losses / len(test_dataloader): .3f} | Accuracy: {accuracy}%')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import os

DEVICE = "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])


fake_data = datasets.ImageFolder("training_data", transform=transform)
fake_loader = DataLoader(fake_data, batch_size=4, shuffle=True)

auth_model = models.resnet18(weights=None)
auth_model.fc = nn.Linear(auth_model.fc.in_features, 1)
auth_model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(auth_model.parameters(), lr=0.0005)

for epoch in range(3):
    for imgs, labels in fake_loader:
        labels = labels.float().unsqueeze(1)
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = auth_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Fake Detector Epoch:", epoch+1)

torch.save(auth_model.state_dict(), "models/auth_model.pth")


disease_data = datasets.ImageFolder("disease", transform=transform)
disease_loader = DataLoader(disease_data, batch_size=4, shuffle=True)

class_names = disease_data.classes
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

disease_model = models.resnet18(weights=None)
disease_model.fc = nn.Linear(disease_model.fc.in_features, len(class_names))
disease_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(disease_model.parameters(), lr=0.0005)

for epoch in range(5):
    for imgs, labels in disease_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = disease_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Disease Model Epoch:", epoch+1)

torch.save(disease_model.state_dict(), "models/best_disease_model.pth")

print("Training Completed. Models Saved Permanently.")


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json
import os
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


AUTH_EPOCHS     = 15
DISEASE_EPOCHS  = 20
BATCH_SIZE      = 16
LR              = 0.0001
WEIGHT_DECAY    = 1e-4
PATIENCE        = 4        
VAL_SPLIT       = 0.2      


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

os.makedirs("models", exist_ok=True)


def split_dataset(dataset, val_split=0.2):
    val_size  = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size],
                        generator=torch.Generator().manual_seed(42))


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            if outputs.shape[1] == 1:
                loss = criterion(outputs, labels.float().unsqueeze(1))
                preds = (torch.sigmoid(outputs) >= 0.5).squeeze().long()
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, save_path, model_name):

    best_val_acc  = 0.0
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)

            if outputs.shape[1] == 1:
                loss = criterion(outputs, labels.float().unsqueeze(1))
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best model saved (Val Acc: {val_acc:.2%})")
            patience_count = 0
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n{model_name} training done. Best Val Acc: {best_val_acc:.2%}\n")
    return best_val_acc


print("=" * 60)
print("TRAINING FAKE IMAGE DETECTOR")
print("=" * 60)

full_auth_data = datasets.ImageFolder("training_data", transform=train_transform)
print(f"Auth dataset: {len(full_auth_data)} images, classes: {full_auth_data.classes}")

class_counts = [0] * len(full_auth_data.classes)
for _, label in full_auth_data:
    class_counts[label] += 1
for name, count in zip(full_auth_data.classes, class_counts):
    print(f"  {name}: {count} images")

train_auth, val_auth = split_dataset(full_auth_data, VAL_SPLIT)

val_auth_data = datasets.ImageFolder("training_data", transform=val_transform)
_, val_auth_indices = split_dataset(
    torch.utils.data.Subset(val_auth_data, range(len(val_auth_data))),
    VAL_SPLIT
)

train_auth_loader = DataLoader(train_auth, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_auth_loader   = DataLoader(val_auth,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

auth_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
auth_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(auth_model.fc.in_features, 1)
)
auth_model.to(DEVICE)

criterion_auth = nn.BCEWithLogitsLoss()
optimizer_auth = optim.Adam(auth_model.parameters(),
                            lr=LR, weight_decay=WEIGHT_DECAY)

auth_acc = train_model(
    auth_model, train_auth_loader, val_auth_loader,
    criterion_auth, optimizer_auth,
    AUTH_EPOCHS, "models/auth_model.pth", "Fake Detector"
)


print("=" * 60)
print("TRAINING DISEASE CLASSIFIER")
print("=" * 60)

full_disease_data = datasets.ImageFolder("disease", transform=train_transform)
class_names = full_disease_data.classes
print(f"Disease dataset: {len(full_disease_data)} images, {len(class_names)} classes")
for name in class_names:
    print(f"  {name}")

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

train_disease, val_disease = split_dataset(full_disease_data, VAL_SPLIT)

train_disease_loader = DataLoader(train_disease, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_disease_loader   = DataLoader(val_disease,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

disease_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
disease_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(disease_model.fc.in_features, len(class_names))
)
disease_model.to(DEVICE)

criterion_disease = nn.CrossEntropyLoss()
optimizer_disease = optim.Adam(disease_model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)

disease_acc = train_model(
    disease_model, train_disease_loader, val_disease_loader,
    criterion_disease, optimizer_disease,
    DISEASE_EPOCHS, "models/best_disease_model.pth", "Disease Classifier"
)


print("=" * 60)
print("ALL TRAINING COMPLETE")
print(f"  Fake Detector Best Val Acc:      {auth_acc:.2%}")
print(f"  Disease Classifier Best Val Acc: {disease_acc:.2%}")
print("  Models saved to /models/")
print("=" * 60)

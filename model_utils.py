import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision import models

def build_model(num_classes, dropout=0.3):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    return model

def build_efficientnet(num_classes, dropout=0.3):
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

def build_densenet(num_classes, dropout=0.3):
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader.dataset), correct / total

def validate_model(model, loader, device, criterion):
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    avg_loss = val_loss / len(loader.dataset)
    return acc, val_labels, val_preds, avg_loss
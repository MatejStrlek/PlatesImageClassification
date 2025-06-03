import torch
from sklearn.metrics import accuracy_score
from torchvision import models
from data_loader import create_dataloaders

CSV_PATH = 'plates_dataset/plates.csv'
IMAGE_SIZE = (224, 128)
BATCH_SIZE = 16
NUM_EPOCHS = 15 # Epoch [15/15], Loss: 0.2310, Accuracy: 0.9336
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset and create dataloaders
train_loader, valid_loader, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, BATCH_SIZE)
num_classes = len(label_map)

# Build model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

"""
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch_idx, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # ✅ Print every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'resnet50_license_plate_model.pth')
"""

# Load the model
model.load_state_dict(torch.load('resnet50_license_plate_model.pth', map_location=DEVICE))
val_preds = []
val_labels = []

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_preds.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

val_acc = accuracy_score(val_labels, val_preds)
print(f"🧪 Validation Accuracy: {val_acc:.4f}")
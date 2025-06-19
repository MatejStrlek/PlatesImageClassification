import torch
import torch.nn as nn
from data_loader import create_dataloaders
from model_utils import build_model, train_one_epoch, validate_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

CSV_PATH = 'plates_dataset/plates.csv'
IMAGE_SIZE = (224, 128)
BATCH_SIZE = 32
NUM_EPOCHS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, BATCH_SIZE)
num_classes = len(label_map)

model = build_model(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
best_val_acc = 0.0

train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    val_acc, val_labels, val_preds, val_loss = validate_model(model, valid_loader, DEVICE, criterion)
    print(f"Validation accuracy: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("\nClassification report (per-class precision, recall, F1):")
    print(classification_report(val_labels, val_preds, target_names=list(label_map.values())))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved new best model with val accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), 'resnet50_license_plate_model.pth')

# Plot and save loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()
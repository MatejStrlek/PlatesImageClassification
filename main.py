import torch
import torch.nn as nn
from data_loader import create_dataloaders
from model_utils import build_model, train_one_epoch, validate_model, plot_confusion_matrix
from sklearn.metrics import classification_report

CSV_PATH = 'plates_dataset/plates.csv'
IMAGE_SIZE = (224, 128)
BATCH_SIZE = 32
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, valid_loader, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, BATCH_SIZE)
num_classes = len(label_map)

# Build model
model = build_model(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
best_val_acc = 0.0

# Train and evaluate
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    val_acc, val_labels, val_preds = validate_model(model, valid_loader, DEVICE)
    print(f"Validation Accuracy: {val_acc:.4f}")
    plot_confusion_matrix(val_labels, val_preds, label_map)

    # Classification report
    print("\n📊 Classification Report (Per-Class Precision, Recall, F1):")
    print(classification_report(val_labels, val_preds, target_names=list(label_map.values())))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved new best model with val accuracy: {val_acc:.4f}")

# Final model save
torch.save(model.state_dict(), 'resnet50_license_plate_model.pth')
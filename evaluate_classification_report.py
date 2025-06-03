import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_loader import create_dataloaders
from model_utils import build_model, validate_model

# === Config ===
CSV_PATH = 'plates_dataset/plates.csv'
MODEL_PATH = 'best_model.pth'
IMAGE_SIZE = (224, 128)
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load validation data and label map ===
_, valid_loader, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, BATCH_SIZE)

# === Load model ===
model = build_model(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

# === Run validation
val_acc, val_labels, val_preds = validate_model(model, valid_loader, DEVICE)
print(f"✅ Validation Accuracy: {val_acc:.4f}")

# === Print and save classification report
report = classification_report(val_labels, val_preds, target_names=list(label_map.values()))
print("\n📊 Classification Report:\n", report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.values())
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
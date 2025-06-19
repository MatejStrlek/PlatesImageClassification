import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_loader import create_dataloaders
from grad_cam import GradCAM, show_gradcam_on_image
from model_utils import build_model, validate_model

CSV_PATH = 'plates_dataset/plates.csv'
MODEL_PATH = 'best_model.pth'
IMAGE_SIZE = (224, 128)
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, valid_loader, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, BATCH_SIZE)

model = build_model(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

val_acc, val_labels, val_preds, _ = validate_model(model, valid_loader, DEVICE, criterion=None)
print(f"Validation accuracy: {val_acc:.4f}")

report = classification_report(val_labels, val_preds, target_names=list(label_map.values()))
with open("classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.values())
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
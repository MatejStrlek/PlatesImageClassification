import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from utils.model_utils import build_model
from model.data_loader import create_dataloaders

# === Config ===
CSV_PATH = '../plates_dataset/plates.csv'
IMAGE_SIZE = (224, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../best_model.pth'

# === Load label map using your existing loader ===
_, _, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, batch_size=32)

# === Load trained model ===
model = build_model(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load test entries from CSV ===
df = pd.read_csv(CSV_PATH)
df = df[df['data set'] == 'test'].copy()
df['filepaths'] = df['filepaths'].apply(lambda p: os.path.join('../plates_dataset', p))

# === Predict on all test images ===
results = []

for _, row in df.iterrows():
    image_path = row['filepaths']
    true_label = row['labels']

    if not os.path.exists(image_path):
        print(f"Missing file: {image_path}")
        continue

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)

    predicted_label = list(label_map.values())[top_class.item()]
    confidence = top_prob.item()

    print(f"{image_path}")
    print(f"Predicted: {predicted_label} | Actual: {true_label} | Confidence: {confidence:.2%}")
    print("—" * 60)

    results.append({
        'image': image_path,
        'actual_label': true_label,
        'predicted_label': predicted_label,
        'confidence': confidence
    })

# Save predictions to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('test_predictions.csv', index=False)
print("\nAll test predictions saved to test_predictions.csv")
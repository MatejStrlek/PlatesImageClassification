import argparse

import torch
import torch.nn.functional as f
from PIL import Image
from torchvision import transforms

from data_loader import create_dataloaders
from model_utils import build_model

# To test this, you need to have a trained model saved as "best_model.pth" and type command:
# python predict_one_plate.py --image path_to_your_image.jpg
def predict_image(image_path, model, label_map, image_size=(224, 128), device='cpu'):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = f.softmax(output, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)

    predicted_label = list(label_map.values())[top_class.item()]
    confidence = top_prob.item()

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_label} ({confidence:.2%} confidence)")

    return predicted_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image to predict")
    args = parser.parse_args()

    # === Settings ===
    CSV_PATH = 'plates_dataset/plates.csv'
    IMAGE_SIZE = (224, 128)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load label map
    _, _, label_map = create_dataloaders(CSV_PATH, IMAGE_SIZE, batch_size=32)

    # Load model
    model = build_model(num_classes=len(label_map))
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)

    # Predict image
    predict_image(args.image, model, label_map, IMAGE_SIZE, device=DEVICE)
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Class labels
CLASS_NAMES = [
    "Normal",
    "Bacterial Pneumonia",
    "Viral Pneumonia"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 3)

    state_dict = torch.load(
        "densenet121_pneumonia.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🔒 STRICT X-RAY VALIDATION
def is_valid_xray(image):
    img = np.array(image)

    # 1️⃣ Must be almost grayscale (RGB channels very similar)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if np.mean(np.abs(r - g)) > 15 or np.mean(np.abs(r - b)) > 15:
        return False

    # 2️⃣ Contrast check (X-rays have mid-level contrast)
    contrast = img.std()
    if contrast < 25 or contrast > 90:
        return False

    # 3️⃣ Brightness check (avoid dark / colorful images)
    brightness = img.mean()
    if brightness < 50 or brightness > 200:
        return False

    return True


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # ❌ Reject non–X-ray images
    if not is_valid_xray(image):
        return "Invalid Image", 0.0

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    return CLASS_NAMES[predicted_class.item()], confidence.item() * 100

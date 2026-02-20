import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# Configuration
# ----------------------------

MODEL_PATH = "models/pcb_defect_resnet18.pth"

CLASS_NAMES = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ----------------------------
# Load Model Once (Global)
# ----------------------------

def load_model():
    model = models.resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


MODEL = load_model()


# ----------------------------
# Image Processing
# ----------------------------

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(DEVICE)


# ----------------------------
# Prediction Function
# ----------------------------

def predict(image_path):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[preds.item()]
    predicted_confidence = confidence.item()

    return predicted_class, float(predicted_confidence)
# ============================================================
# BIRD SPECIES PREDICTION
# ============================================================
# Usage: Change IMAGE_PATH below to any bird photo and run
# ============================================================

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# ============================================================
# CONFIGURATION - CHANGE THESE
# ============================================================

# Path to the image you want to classify
IMAGE_PATH = r"C:\Users\Doruk\Downloads\original.jpg"  # <--- CHANGE THIS

# Which model to use for prediction
MODEL_NAME = 'mobilenet_v2'  # Match the model you trained

# Path to the saved model
MODEL_PATH = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\models\mobilenet_v2_best.pth"

# Path to label mapping
LABEL_MAP_PATH = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\label_mapping.csv"

NUM_CLASSES = 23

# ============================================================
# LOAD LABEL MAPPING
# ============================================================

label_map_df = pd.read_csv(LABEL_MAP_PATH)
label_to_species = dict(zip(label_map_df['label'], label_map_df['species']))

print(f"\nLoaded {len(label_to_species)} species")

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(model_name, model_path, num_classes):
    """Load a trained model from .pth file."""

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=None)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded from: {model_path}")
    print(f"Best accuracy during training: {checkpoint.get('best_acc', 'N/A')}")

    return model


model = load_model(MODEL_NAME, MODEL_PATH, NUM_CLASSES)

# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path, model_name):
    """Load and preprocess an image for prediction."""

    if model_name == 'inception_v3':
        img_size = 299
        resize_size = 342
    else:
        img_size = 224
        resize_size = 256

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


# ============================================================
# PREDICT
# ============================================================

def predict(model, image_path, model_name, label_to_species, device, top_k=5):
    """Predict the species of a bird in an image."""

    # Prepare image
    image_tensor = prepare_image(image_path, model_name)
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    # Convert to readable results
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    print(f"\n{'=' * 60}")
    print(f"  IMAGE: {os.path.basename(image_path)}")
    print(f"{'=' * 60}")
    print(f"\n  Top {top_k} Predictions:")
    print(f"  {'─' * 50}")

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        species = label_to_species[idx]
        bar = '█' * int(prob * 30)
        marker = " <<<" if i == 0 else ""
        print(f"  {i+1}. {species:<35s} {prob*100:5.2f}% {bar}{marker}")

    print(f"\n  Predicted species: {label_to_species[top_indices[0]]}")
    print(f"  Confidence: {top_probs[0]*100:.2f}%")

    return label_to_species[top_indices[0]], top_probs[0]


# ============================================================
# RUN PREDICTION
# ============================================================

if __name__ == '__main__':

    if not os.path.exists(IMAGE_PATH):
        print(f"\nERROR: Image not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH at the top of this script.")
    else:
        species, confidence = predict(
            model, IMAGE_PATH, MODEL_NAME, label_to_species, device
        )
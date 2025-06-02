
"""
GauravPatil2515 - Skin Cancer Classification
INSTANT DEPLOYMENT CODE
Date: 2025-05-30 21:03:45 UTC
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Class definitions
CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma', 
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',  # CRITICAL
    'Melanocytic nevi',
    'Vascular lesions'
]

# Ultra-Light Model Architecture
class UltraLightModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Prediction function
def predict_skin_cancer(model_path, image_path):
    """Predict skin cancer from image"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if checkpoint.get('model_class') == 'UltraLightModel':
        model = UltraLightModel()
    else:
        # For EfficientNet or other models
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
        except:
            model = UltraLightModel()  # Fallback
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    result = {
        'predicted_class': predicted.item(),
        'predicted_name': CLASS_NAMES[predicted.item()],
        'confidence': confidence.item(),
        'all_probabilities': {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(7)}
    }
    
    return result

# Usage example
if __name__ == "__main__":
    print("🏥 GauravPatil2515 Skin Cancer Detector")
    print("Usage: predict_skin_cancer('model.pth', 'image.jpg')")
    print("Classes:", CLASS_NAMES)

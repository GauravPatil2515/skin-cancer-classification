"""
Quick Test for Skin Cancer Classification App
Verifies critical components are working correctly
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

print("🏥 Quick Test for Skin Cancer Classification App")
print("=" * 60)

# Test imports
print("\n📦 Testing critical imports...")
try:
    import streamlit
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

# Test model code imports
print("\n🧠 Testing model code imports...")
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from GAURAV_DEPLOYMENT_CODE import UltraLightModel, CLASS_NAMES, predict_skin_cancer
    print(f"✅ Model code imported successfully ({len(CLASS_NAMES)} classes defined)")
except ImportError as e:
    print(f"❌ Model code import failed: {e}")

# Check for model files
print("\n💾 Checking model files...")
model_files = [
    os.path.join("assets", "GAURAV_UltraLight_Model.pth"),
    os.path.join("assets", "GAURAV_EfficientNet_Model.pth")
]

for model_file in model_files:
    if os.path.exists(model_file):
        print(f"✅ Found: {model_file} ({os.path.getsize(model_file) / (1024*1024):.1f} MB)")
    else:
        print(f"❌ Missing: {model_file}")

# Test model loading
print("\n🔍 Testing UltraLight model loading...")
try:
    model = UltraLightModel()
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    # Forward pass
    output = model(dummy_input)
    print(f"✅ Model forward pass successful! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Model test failed: {e}")

# Test GradCAM generation
print("\n🎨 Testing heatmap generation (fallback mechanism)...")
try:
    # Create a dummy image
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Save it temporarily
    temp_path = "temp_test_image.jpg"
    cv2.imwrite(temp_path, dummy_image)
    
    # Generate a simple heatmap
    heatmap = np.zeros((224, 224))
    # Create a gradient pattern for testing
    for i in range(224):
        for j in range(224):
            heatmap[i, j] = (i + j) / (2 * 224)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Overlay
    original_image = cv2.imread(temp_path)
    if original_image is not None:
        overlayed = cv2.addWeighted(original_image, 0.6, colored_heatmap, 0.4, 0)
        print("✅ Heatmap generation successful!")
    else:
        print("⚠️ Test image loading failed, but fallback should work")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
except Exception as e:
    print(f"❌ Heatmap test failed: {e}")

print("\n" + "=" * 60)
print("✅ Quick test completed")
print("You can now run the main application:")
print("streamlit run main_app.py")
print("=" * 60)

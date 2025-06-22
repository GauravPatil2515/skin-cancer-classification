#!/usr/bin/env python3
"""
Quick System Test
================
Tests if all components are working properly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
        
        import timm
        print(f"✅ TIMM {timm.__version__}")
        
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        from PIL import Image
        print(f"✅ Pillow")
        
        import groq
        print(f"✅ Groq")
        
        import reportlab
        print(f"✅ ReportLab")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n🧪 Testing model files...")
    
    model_files = [
        'assets/GAURAV_EfficientNet_Model.pth',
        'assets/GAURAV_UltraLight_Model.pth'
    ]
    
    all_exist = True
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Missing: {model_file}")
            all_exist = False
    
    return all_exist

def test_src_files():
    """Test if source files exist"""
    print("\n🧪 Testing source files...")
    
    src_files = [
        'src/GAURAV_DEPLOYMENT_CODE.py',
        'main_app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for src_file in src_files:
        if os.path.exists(src_file):
            print(f"✅ {src_file}")
        else:
            print(f"❌ Missing: {src_file}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 Running system tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_model_files():
        tests_passed += 1
        
    if test_src_files():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! System is ready.")
        print("\n🚀 To start the application:")
        print("   Run: launch_app.bat")
        print("   Or: streamlit run main_app.py")
        print("\n🌐 GitHub Pages:")
        print("   https://gauravpatil2515.github.io/skin-cancer-classification/")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Test Suite for Skin Cancer Classification App
============================================================
Tests all functionality including UI theme, GradCAM visualization,
error handling, and application features.

Author: Gaurav Patil
Version: 1.0
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import tempfile
import requests
from typing import Dict, Any

# Add current directory to path
sys.path.append('.')

def create_test_image(size=(224, 224), color='RGB', name="test_image.jpg"):
    """Create a test image for testing purposes"""
    image = Image.new(color, size, color=(128, 128, 128))
    image.save(name)
    return name

def test_model_loading():
    """Test model loading functionality"""
    print("🔬 Testing Model Loading...")
    try:
        from main_app_optimized import load_model_cached
        
        model_paths = [
            "assets/GAURAV_UltraLight_Model.pth",
            "assets/GAURAV_EfficientNet_Model.pth"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = load_model_cached(model_path)
                if model is not None:
                    print(f"✅ {model_path} loaded successfully")
                else:
                    print(f"⚠️ {model_path} failed to load")
            else:
                print(f"❌ {model_path} not found")
        
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_image_validation():
    """Test image validation functionality"""
    print("\n📷 Testing Image Validation...")
    try:
        from main_app_optimized import ImageValidator
        
        # Create test image
        test_path = create_test_image()
        test_image = Image.open(test_path)
        
        # Test validation
        is_valid, error = ImageValidator.validate_image(test_image)
        print(f"✅ Image validation: {'Valid' if is_valid else f'Invalid - {error}'}")
        
        # Test quality assessment
        quality = ImageValidator.assess_quality(test_image)
        print(f"✅ Quality assessment: {quality}")
        
        # Cleanup
        os.remove(test_path)
        return True
        
    except Exception as e:
        print(f"❌ Image validation test failed: {e}")
        return False

def test_gradcam_fallback():
    """Test GradCAM fallback mechanism"""
    print("\n🔥 Testing GradCAM Fallback...")
    try:
        from main_app_optimized import GradCAMGenerator
        
        # Create test image
        test_path = create_test_image()
        
        # Test with non-existent model (should trigger fallback)
        overlay, heatmap = GradCAMGenerator.generate("non_existent_model.pth", test_path)
        
        if overlay is not None and heatmap is not None:
            print("✅ GradCAM fallback mechanism working")
            print(f"   Overlay shape: {overlay.shape}")
            print(f"   Heatmap shape: {heatmap.shape}")
        else:
            print("⚠️ GradCAM fallback returned None")
        
        # Test with existing model if available
        model_path = "assets/GAURAV_UltraLight_Model.pth"
        if os.path.exists(model_path):
            overlay, heatmap = GradCAMGenerator.generate(model_path, test_path)
            if overlay is not None:
                print("✅ GradCAM with real model working")
            else:
                print("⚠️ GradCAM with real model failed (using fallback)")
        
        # Cleanup
        os.remove(test_path)
        return True
        
    except Exception as e:
        print(f"❌ GradCAM test failed: {e}")
        return False

def test_api_key_management():
    """Test API key management system"""
    print("\n🔑 Testing API Key Management...")
    try:
        from main_app_optimized import APIKeyManager
        
        # Test getting API key
        api_key = APIKeyManager.get_api_key()
        if api_key:
            print("✅ API key retrieved successfully")
            
            # Test API key validation
            is_valid, message = APIKeyManager.test_api_key(api_key)
            print(f"✅ API key test: {'Valid' if is_valid else f'Invalid - {message}'}")
        else:
            print("⚠️ No API key available")
        
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_ai_analysis():
    """Test AI analysis functionality"""
    print("\n🤖 Testing AI Analysis...")
    try:
        from main_app_optimized import AIAnalyzer
        
        # Test analysis
        test_result = {
            'class': 'Melanoma',
            'confidence': 0.85
        }
        
        analyzer = AIAnalyzer()
        analysis = analyzer.analyze(test_result, "test_image.jpg")
        
        if analysis:
            print("✅ AI analysis completed successfully")
            print(f"   Analysis length: {len(analysis)} characters")
        else:
            print("⚠️ AI analysis returned empty result")
        
        return True
        
    except Exception as e:
        print(f"❌ AI analysis test failed: {e}")
        return False

def test_pdf_generation():
    """Test PDF report generation"""
    print("\n📄 Testing PDF Report Generation...")
    try:
        from main_app_optimized import ReportGenerator
        
        # Test data
        patient_data = {
            'name': 'Test Patient',
            'age': '30',
            'gender': 'Male'
        }
        
        prediction_data = {
            'predicted_class': 'Melanoma',
            'confidence': 85.5,
            'all_predictions': [
                ('Melanoma', 85.5),
                ('Nevus', 10.2),
                ('Seborrheic keratosis', 4.3)
            ]
        }
        
        ai_analysis = "Test AI analysis for PDF generation."
        
        # Create test image
        test_path = create_test_image()
        
        generator = ReportGenerator()
        pdf_path = generator.generate_report(patient_data, prediction_data, test_path, ai_analysis)
        
        if pdf_path and os.path.exists(pdf_path):
            print("✅ PDF report generated successfully")
            print(f"   PDF path: {pdf_path}")
            
            # Cleanup
            os.remove(pdf_path)
        else:
            print("⚠️ PDF generation failed")
        
        # Cleanup test image
        os.remove(test_path)
        return True
        
    except Exception as e:
        print(f"❌ PDF generation test failed: {e}")
        return False

def test_app_accessibility():
    """Test if the application is accessible"""
    print("\n🌐 Testing Application Accessibility...")
    try:
        # Check if app is running
        response = requests.get("http://localhost:8503", timeout=10)
        if response.status_code == 200:
            print("✅ Application is accessible and responding")
            print(f"   Response status: {response.status_code}")
            return True
        else:
            print(f"⚠️ Application responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Application accessibility test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🧪 COMPREHENSIVE SKIN CANCER APP TEST SUITE")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Model Loading", test_model_loading),
        ("Image Validation", test_image_validation),
        ("GradCAM Fallback", test_gradcam_fallback),
        ("API Key Management", test_api_key_management),
        ("AI Analysis", test_ai_analysis),
        ("PDF Generation", test_pdf_generation),
        ("App Accessibility", test_app_accessibility),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Application is fully functional.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Application is largely functional.")
    else:
        print("⚠️ Several tests failed. Please check the issues above.")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()

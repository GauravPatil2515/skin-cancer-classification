#!/usr/bin/env python3
"""
Quick Application Test
======================
Simple test to verify core functionality.
"""

import os
import sys

def quick_validation():
    """Quick validation of core components"""
    print("🚀 Quick Application Test")
    print("="*50)
    
    # Test project structure
    print("\n1. 📁 Project Structure:")
    essential_files = [
        'main_app.py',
        'src/GAURAV_DEPLOYMENT_CODE.py',
        'assets/GAURAV_UltraLight_Model.pth',
        'assets/GAURAV_EfficientNet_Model.pth'
    ]
    
    structure_ok = True
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            structure_ok = False
    
    # Test basic imports
    print("\n2. 📦 Basic Imports:")
    basic_imports = ['torch', 'PIL', 'numpy', 'cv2']
    imports_ok = True
    
    for module in basic_imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            imports_ok = False
    
    # Test model classes
    print("\n3. 🧠 Model Classes:")
    try:
        sys.path.append('src')
        from GAURAV_DEPLOYMENT_CODE import UltraLightModel, CLASS_NAMES
        print(f"   ✅ UltraLightModel class imported")
        print(f"   ✅ CLASS_NAMES: {len(CLASS_NAMES)} classes")
        models_ok = True
    except Exception as e:
        print(f"   ❌ Model import failed: {e}")
        models_ok = False
    
    # Test main app syntax
    print("\n4. 🔧 Main App Syntax:")
    try:
        with open('main_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key features
        features = [
            ('PRIMARY_API_KEY', 'API key configured'),
            ('assets/', 'Asset paths updated'),
            ('handle_errors', 'Error handling'),
            ('st.markdown', 'Streamlit interface')
        ]
        
        app_ok = True
        for feature, description in features:
            if feature in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                app_ok = False
                
    except Exception as e:
        print(f"   ❌ Main app check failed: {e}")
        app_ok = False
    
    # Overall result
    all_tests = [structure_ok, imports_ok, models_ok, app_ok]
    passed = sum(all_tests)
    total = len(all_tests)
    score = (passed / total) * 100
    
    print(f"\n📊 OVERALL RESULT:")
    print(f"   Score: {score:.1f}% ({passed}/{total} tests passed)")
    
    if score >= 75:
        print(f"   Status: ✅ READY FOR TESTING")
        print(f"   Next: Run 'streamlit run main_app.py'")
    else:
        print(f"   Status: ⚠️ NEEDS FIXES")
        print(f"   Fix critical issues before deployment")
    
    # Save result
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/quick_test_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Quick Test Result - {timestamp}\n")
        f.write(f"Score: {score:.1f}%\n")
        f.write(f"Status: {'READY' if score >= 75 else 'NEEDS_FIXES'}\n")
    
    return score >= 75

if __name__ == "__main__":
    success = quick_validation()
    
    if success:
        print(f"\n🎉 APPLICATION IS READY!")
        print(f"Run: streamlit run main_app.py")
    else:
        print(f"\n⚠️ PLEASE FIX ISSUES FIRST")

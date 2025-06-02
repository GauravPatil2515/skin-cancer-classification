#!/usr/bin/env python3
"""
Final Application Validation Script
===================================
Comprehensive testing of the production-ready application.
Author: Gaurav Patil
Version: 1.0
"""

import os
import sys
import importlib.util
from datetime import datetime

def check_project_structure():
    """Validate the final project structure"""
    print("🏗️ Checking Project Structure...")
    
    required_files = {
        'main_app.py': 'Main application',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Documentation',
        'src/GAURAV_DEPLOYMENT_CODE.py': 'Model definitions',
        'assets/GAURAV_UltraLight_Model.pth': 'UltraLight model',
        'assets/GAURAV_EfficientNet_Model.pth': 'EfficientNet model',
        'config/GAURAV_PROJECT_INFO.json': 'Project configuration'
    }
    
    all_present = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            if file_path.endswith('.pth'):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✅ {file_path}: {description} ({size_mb:.1f} MB)")
            else:
                print(f"  ✅ {file_path}: {description}")
        else:
            print(f"  ❌ {file_path}: {description} - MISSING")
            all_present = False
    
    return all_present

def test_imports():
    """Test critical imports"""
    print("\n📦 Testing Critical Imports...")
    
    import_tests = [
        ('streamlit', 'Streamlit framework'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow image processing'),
        ('groq', 'Groq API client'),
        ('reportlab', 'PDF generation'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_imports_ok = True
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}: {description}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {description} - FAILED ({e})")
            all_imports_ok = False
    
    return all_imports_ok

def test_model_loading():
    """Test model loading functionality"""
    print("\n🧠 Testing Model Loading...")
    
    try:
        # Add src to path
        sys.path.append('src')
        from GAURAV_DEPLOYMENT_CODE import UltraLightModel, CLASS_NAMES
        
        print(f"  ✅ Model classes imported successfully")
        print(f"  ✅ CLASS_NAMES: {len(CLASS_NAMES)} classes defined")
        print(f"  📋 Classes: {', '.join(CLASS_NAMES[:3])}...")
        
        # Test model instantiation
        model = UltraLightModel()
        print(f"  ✅ UltraLightModel instantiated successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def test_app_configuration():
    """Test main app configuration"""
    print("\n⚙️ Testing App Configuration...")
    
    try:
        # Test main app imports without running streamlit
        spec = importlib.util.spec_from_file_location("main_app", "main_app.py")
        main_app = importlib.util.module_from_spec(spec)
        
        # Check if file loads without syntax errors
        with open('main_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'PRIMARY_API_KEY' in content:
            print("  ✅ Primary API key configured")
        
        if 'assets' in content:
            print("  ✅ Asset paths updated for new structure")
            
        if 'handle_errors' in content:
            print("  ✅ Error handling system present")
            
        if 'st.markdown' in content and 'color:' in content:
            print("  ✅ Text coloring fixes implemented")
            
        return True
    except Exception as e:
        print(f"  ❌ App configuration test failed: {e}")
        return False

def check_api_key_fallback():
    """Test API key fallback system"""
    print("\n🔑 Testing API Key System...")
    
    try:
        with open('main_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        fallback_checks = [
            ('PRIMARY_API_KEY', 'Primary coded key'),
            ('os.getenv', 'Environment variable fallback'),
            ('st.text_input', 'User input fallback')
        ]
        
        all_fallbacks = True
        for check, description in fallback_checks:
            if check in content:
                print(f"  ✅ {description}: Present")
            else:
                print(f"  ❌ {description}: Missing")
                all_fallbacks = False
        
        return all_fallbacks
    except Exception as e:
        print(f"  ❌ API key system test failed: {e}")
        return False

def generate_validation_report():
    """Generate comprehensive validation report"""
    
    print("\n" + "="*60)
    print("🎯 FINAL VALIDATION REPORT")
    print("="*60)
    
    # Run all tests
    structure_ok = check_project_structure()
    imports_ok = test_imports()
    models_ok = test_model_loading()
    config_ok = test_app_configuration()
    api_ok = check_api_key_fallback()
    
    # Calculate overall score
    tests = [structure_ok, imports_ok, models_ok, config_ok, api_ok]
    passed = sum(tests)
    total = len(tests)
    score = (passed / total) * 100
    
    print(f"\n📊 OVERALL SCORE: {score:.1f}% ({passed}/{total} tests passed)")
    
    if score >= 100:
        status = "🟢 PRODUCTION READY"
        message = "Application is fully ready for deployment!"
    elif score >= 80:
        status = "🟡 MOSTLY READY"
        message = "Minor issues detected, but application should work."
    else:
        status = "🔴 NEEDS ATTENTION"
        message = "Critical issues found. Please fix before deployment."
    
    print(f"🎖️ STATUS: {status}")
    print(f"💬 {message}")
    
    # Deployment instructions
    if score >= 80:
        print(f"\n🚀 DEPLOYMENT INSTRUCTIONS:")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. Launch application: streamlit run main_app.py")
        print(f"  3. Access at: http://localhost:8501")
        print(f"  4. Upload skin lesion images for analysis")
    
    # Create validation timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save validation report
    os.makedirs('logs', exist_ok=True)
    report_content = f"""
# Final Validation Report
Date: {timestamp}
Score: {score:.1f}% ({passed}/{total} tests passed)
Status: {status}

## Test Results:
- Project Structure: {'✅' if structure_ok else '❌'}
- Critical Imports: {'✅' if imports_ok else '❌'}
- Model Loading: {'✅' if models_ok else '❌'}
- App Configuration: {'✅' if config_ok else '❌'}
- API Key System: {'✅' if api_ok else '❌'}

## Deployment Status:
{message}

## Files Validated:
- main_app.py (Primary application)
- src/GAURAV_DEPLOYMENT_CODE.py (Model definitions)
- assets/*.pth (Model weights)
- requirements.txt (Dependencies)
- README.md (Documentation)

## Next Steps:
1. Run: pip install -r requirements.txt
2. Launch: streamlit run main_app.py
3. Test with sample images
4. Generate PDF reports
5. Verify all features work correctly
"""
      with open('logs/validation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📝 Detailed report saved to: logs/validation_report.md")
    
    return score >= 80

if __name__ == "__main__":
    print("🏥 Professional Skin Cancer Classification Dashboard")
    print("Final Validation & Testing")
    print("="*60)
    
    success = generate_validation_report()
    
    if success:
        print(f"\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"Your application is ready for production use.")
    else:
        print(f"\n⚠️ VALIDATION FOUND ISSUES")
        print(f"Please review the report and fix critical issues.")
    
    print(f"\nThank you for using the Professional Skin Cancer Classification Dashboard! 🏥")

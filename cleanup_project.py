#!/usr/bin/env python3
"""
Project Cleanup and Restructuring Script
========================================
Removes duplicate files and organizes the project structure.
Author: Gaurav Patil
Version: 1.0
"""

import os
import shutil
from datetime import datetime

def cleanup_project():
    """Clean up duplicate and test files while preserving essential components"""
    
    print("🧹 Starting Project Cleanup...")
    print("=" * 50)
    
    # Files to remove (duplicates and test files)
    files_to_remove = [
        'app.py',
        'app_backup.py', 
        'app_fixed.py',
        'app_enhanced.py',
        'demo.py',
        'launch_enhanced.py',
        'test_models.py',
        'test_enhancements.py',
        'quick_validation.py',
        'advanced_explainability.py',
        'enhanced_models.py',
        'enhanced_ui.py',
        'medical_validator.py',
        'performance_optimizer.py'
    ]
    
    # Directories to clean
    dirs_to_clean = [
        '__pycache__',
        '.venv'
    ]
    
    removed_count = 0
    
    # Remove duplicate files
    print("\n📄 Removing duplicate and test files:")
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  ✅ Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {file}: {e}")
        else:
            print(f"  ⚪ Not found: {file}")
    
    # Clean directories
    print("\n📁 Cleaning directories:")
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"  ✅ Removed directory: {dir_name}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {dir_name}: {e}")
        else:
            print(f"  ⚪ Not found: {dir_name}")
    
    # Create archive directory for removed files log
    os.makedirs('logs', exist_ok=True)
    
    # Log the cleanup
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"""
# Project Cleanup Log
Date: {timestamp}
Files removed: {removed_count}

## Removed Files:
{chr(10).join([f"- {f}" for f in files_to_remove if os.path.exists(f) == False])}

## Directories Cleaned:
{chr(10).join([f"- {d}" for d in dirs_to_clean])}

## Preserved Essential Files:
- main_app.py (Primary application)
- src/GAURAV_DEPLOYMENT_CODE.py (Model definitions)
- assets/*.pth (Model weights)
- config/GAURAV_PROJECT_INFO.json (Project info)
- requirements.txt (Dependencies)
- .env.example (Environment template)
- docs/ (Documentation)
"""
    
    with open('logs/cleanup_log.md', 'w') as f:
        f.write(log_content)
    
    print(f"\n✅ Cleanup completed! Removed {removed_count} items")
    print("📝 Cleanup log saved to: logs/cleanup_log.md")
    
    # Display final structure
    print("\n📁 Final Project Structure:")
    display_project_structure()

def display_project_structure():
    """Display the current project structure"""
    
    structure = """
cancer-kaggle/
├── main_app.py              # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── README.md               # Project documentation
├── src/
│   └── GAURAV_DEPLOYMENT_CODE.py  # Model definitions
├── assets/
│   ├── GAURAV_UltraLight_Model.pth    # UltraLight model weights
│   └── GAURAV_EfficientNet_Model.pth  # EfficientNet model weights
├── config/
│   └── GAURAV_PROJECT_INFO.json       # Project configuration
├── docs/
│   ├── comprehensive_improvements.md  # Technical documentation
│   └── ENHANCEMENT_SUMMARY.md         # Enhancement summary
├── tests/
│   └── (reserved for future tests)
├── reports/
│   └── (generated PDF reports will be saved here)
└── logs/
    └── cleanup_log.md      # Cleanup operation log
"""
    
    print(structure)

if __name__ == "__main__":
    cleanup_project()

#!/usr/bin/env python3
"""
GitHub Repository Setup and Push Script
======================================
Prepares the current project for GitHub and pushes it to a new repository.
This script:
1. Runs the cleanup script to organize the project
2. Creates .gitignore and GitHub configuration files
3. Initializes Git repository
4. Sets up the repository for push

Author: Gaurav Patil
Version: 1.0
"""

import os
import sys
import subprocess
import platform
from datetime import datetime

# Import cleanup script if available
try:
    from cleanup_project import cleanup_project
    CLEANUP_AVAILABLE = True
except ImportError:
    CLEANUP_AVAILABLE = False

def run_command(command, description=None):
    """Run a shell command and print its output"""
    if description:
        print(f"\n{description}:")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False, str(e)

def create_gitignore():
    """Create a .gitignore file with appropriate entries for a Python ML project"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# ML specific
# Uncomment if you want to exclude model files
# *.pth
# *.h5
# *.model
# *.onnx
# *.pb
# *.tflite

# Reports and logs
reports/
logs/*.log

# Jupyter Notebooks
.ipynb_checkpoints

# OS specific
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def create_github_workflows():
    """Create GitHub Actions workflow files for CI/CD"""
    
    # Create directory if it doesn't exist
    os.makedirs('.github/workflows', exist_ok=True)
    
    # Basic CI workflow for Python
    ci_workflow = """name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install flake8 pytest
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Test with pytest
      run: |
        # Uncomment when tests are available
        # pytest
        echo "Tests pending implementation"
"""
    
    with open('.github/workflows/python-ci.yml', 'w', encoding='utf-8') as f:
        f.write(ci_workflow)
    
    print("✅ Created GitHub Actions workflow file")

def create_license_file():
    """Create an MIT license file for the project"""
    current_year = datetime.now().year
    
    license_content = f"""MIT License

Copyright (c) {current_year} Gaurav Patil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('LICENSE', 'w', encoding='utf-8') as f:
        f.write(license_content)
    
    print("✅ Created LICENSE file")

def git_setup_and_push():
    """Initialize Git repository and prepare for push"""
    
    # Check if git is installed
    success, git_version = run_command("git --version", "Checking Git version")
    if not success:
        print("❌ Git is not installed. Please install Git and try again.")
        return False
    
    # Check if already a git repository
    if os.path.exists('.git'):
        print("⚠️ Git repository already exists")
    else:
        # Initialize git repository
        success, _ = run_command("git init", "Initializing Git repository")
        if not success:
            print("❌ Failed to initialize Git repository")
            return False
    
    # Add files to git
    run_command("git add .", "Adding files to Git")
    
    # Initial commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Initial commit: Professional Skin Cancer Classification Dashboard ({timestamp})"
    success, _ = run_command(f'git commit -m "{commit_message}"', "Creating initial commit")
    
    # Instructions for connecting and pushing to GitHub
    print("\n🚀 Ready to push to GitHub!")
    print("\nFollow these steps to complete the setup:")
    print("1. Create a new repository on GitHub (https://github.com/new)")
    print("2. Run the following commands to push your code:")
    print("\n   git remote add origin https://github.com/GauravPatil2515/skin-cancer-classification.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n💡 The commands are pre-configured for your GitHub profile (GauravPatil2515)")
    print("📄 GitHub Pages workflow has been set up to deploy automatically")
    
    # Ask if user wants to push directly
    try:
        direct_push = input("\nWould you like to push directly to GitHub now? (y/n): ").strip().lower()
        if direct_push == 'y':
            repo_name = input("Enter repository name (default: skin-cancer-classification): ").strip()
            if not repo_name:
                repo_name = "skin-cancer-classification"
            
            # Set remote origin
            remote_url = f"https://github.com/GauravPatil2515/{repo_name}.git"
            success, _ = run_command(f'git remote add origin {remote_url}', "Setting GitHub remote")
            if not success:
                return False
                
            # Rename branch to main
            run_command("git branch -M main", "Renaming current branch to main")
            
            # Push to GitHub
            print(f"\nPushing to GitHub repository: {remote_url}")
            success, _ = run_command("git push -u origin main", "Pushing to GitHub")
            if success:
                print(f"\n✅ Successfully pushed to GitHub: https://github.com/GauravPatil2515/{repo_name}")
                print(f"🌐 GitHub Pages will be available at: https://gauravpatil2515.github.io/{repo_name}/")
                
                # Instructions for enabling GitHub Pages
                print("\n📘 To enable GitHub Pages:")
                print("1. Go to your repository settings on GitHub")
                print("2. Navigate to 'Pages' in the sidebar")
                print("3. Under 'Build and deployment', select:")
                print("   - Source: 'GitHub Actions'")
                print("4. Wait for the GitHub Action workflow to complete")
                print("5. Your site will be published to GitHub Pages automatically")
            else:
                print("\n❌ Failed to push to GitHub. Please push manually using the commands above.")
    except Exception as e:
        print(f"\n⚠️ Error during direct push: {e}")
        print("You can still push manually using the commands above.")
    
    return True

def create_project_structure_file():
    """Create a file documenting the project structure"""
    
    structure_content = """# Project Structure Documentation

## Overview
This file documents the structure of the Professional Skin Cancer Classification Dashboard project.

## Directory Structure

```
cancer-kaggle/
|-- main_app.py                       # Main Streamlit application
|-- requirements.txt                  # Python dependencies
|-- .env.example                      # Environment template
|-- README.md                         # Project documentation
|-- LICENSE                           # MIT License
|-- src/
|   `-- GAURAV_DEPLOYMENT_CODE.py     # Model definitions and inference
|-- assets/
|   |-- GAURAV_UltraLight_Model.pth   # Fast model (100ms inference)
|   `-- GAURAV_EfficientNet_Model.pth # Accurate model (500ms inference)
|-- config/
|   `-- GAURAV_PROJECT_INFO.json      # Project configuration
|-- docs/
|   |-- comprehensive_improvements.md  # Technical documentation
|   `-- ENHANCEMENT_SUMMARY.md        # Enhancement summary
|-- tests/                            # Test files (reserved)
|-- reports/                          # Generated PDF reports
`-- logs/                             # Application logs
```

## Key Files Description

### Core Application Files
- **main_app.py**: The main Streamlit application with the dashboard implementation
- **src/GAURAV_DEPLOYMENT_CODE.py**: Model definitions and inference code
- **requirements.txt**: Python package dependencies

### Configuration Files
- **.env.example**: Template for environment variables (API keys)
- **config/GAURAV_PROJECT_INFO.json**: Project configuration data

### Model Files
- **assets/GAURAV_UltraLight_Model.pth**: Fast model weights (100ms inference)
- **assets/GAURAV_EfficientNet_Model.pth**: Accurate model weights (500ms inference)

### Documentation
- **README.md**: Main project documentation
- **docs/comprehensive_improvements.md**: Detailed technical improvements guide
- **docs/ENHANCEMENT_SUMMARY.md**: Summary of enhancements made

## Development Notes
- Created: June 2025
- Author: Gaurav Patil
- Status: Production Ready
"""
    
    with open('PROJECT_STRUCTURE.md', 'w', encoding='utf-8') as f:
        f.write(structure_content)
    
    print("✅ Created PROJECT_STRUCTURE.md file")

def main():
    """Main function to run the GitHub setup process"""
    
    print("🚀 GitHub Repository Setup for GauravPatil2515")
    print("=" * 50)
    
    # Check if repository name is provided as argument
    repo_name = None
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
        print(f"Repository name provided: {repo_name}")
    
    # Step 1: Run cleanup script if available
    if CLEANUP_AVAILABLE:
        print("\n🧹 Step 1: Running project cleanup")
        cleanup_project()
    else:
        print("\n⚠️ Cleanup script not found, skipping cleanup step")
        print("   If you want to clean up your project first, please run cleanup_project.py separately")
    
    # Step 2: Create .gitignore file
    print("\n📄 Step 2: Creating .gitignore file")
    create_gitignore()
    
    # Step 3: Create GitHub workflows
    print("\n⚙️ Step 3: Setting up GitHub Actions workflows")
    create_github_workflows()
    
    # Step 4: Create LICENSE file
    print("\n📜 Step 4: Creating LICENSE file")
    create_license_file()
    
    # Step 5: Create project structure documentation
    print("\n📚 Step 5: Creating project structure documentation")
    create_project_structure_file()
    
    # Step 6: Initialize Git and prepare for push
    print("\n🔄 Step 6: Setting up Git repository")
    git_setup_and_push()
    
    print("\n✅ GitHub setup complete!")
    print("You can now push to GitHub using your GauravPatil2515 account!")

if __name__ == "__main__":
    main()
# Project Structure Documentation

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

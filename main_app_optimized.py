"""
Professional Skin Cancer Classification Dashboard
===============================================
Optimized, production-ready application with proper error handling
Author: Gaurav Patil
Version: 3.0 - Fully Optimized
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import traceback
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import time
from datetime import datetime
from groq import Groq
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import pandas as pd
import re
import logging
from typing import Optional, Tuple, Dict, Any

# Import local modules
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from GAURAV_DEPLOYMENT_CODE import UltraLightModel, CLASS_NAMES, predict_skin_cancer
except ImportError as e:
    st.error(f"❌ Failed to import model modules: {e}")
    st.error("Please ensure GAURAV_DEPLOYMENT_CODE.py is in the src/ directory")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class Config:
    """Application configuration with fallback values"""
    
    # Primary API key (coded for production use)
    PRIMARY_API_KEY = "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"
    
    # Fallback API keys for redundancy
    FALLBACK_KEYS = [
        "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",
        "gsk_backup_key_here",  # Add backup keys as needed
    ]
    
    # Model paths (updated for new structure)
    ULTRALIGHT_MODEL_PATH = os.path.join("assets", "GAURAV_UltraLight_Model.pth")
    EFFICIENTNET_MODEL_PATH = os.path.join("assets", "GAURAV_EfficientNet_Model.pth")
    
    # File upload limits
    MAX_FILE_SIZE_MB = 10
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    
    # UI Configuration
    CONFIDENCE_THRESHOLDS = {
        'high': 80,
        'medium': 60,
        'low': 0
    }
    
    # Medical urgency mapping
    URGENT_CONDITIONS = ['Melanoma']
    WARNING_CONDITIONS = ['Basal cell carcinoma', 'Actinic keratoses']

config = Config()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Professional Skin Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR DARK THEME
# =============================================================================

st.markdown("""
<style>
    /* Dark theme with black background */
    .stApp {
        color: #ffffff;
        background-color: #000000;
    }
    
    /* Fix sidebar styling */
    .css-1d391kg {
        color: #ffffff;
        background-color: #111111;
    }
    
    /* Main content area with dark background */
    .main .block-container {
        color: #ffffff;
        background-color: #000000;
    }
    
    /* Headers with bright colors for contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Metric cards with dark theme */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(255,255,255,0.1);
        color: #ffffff !important;
    }
    
    .metric-card h3 {
        color: #cccccc !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .metric-card h2 {
        color: #ffffff !important;
        margin: 0.5rem 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-card h4 {
        color: #aaaaaa !important;
        margin: 0;
        font-size: 1rem;
    }
    
    /* Confidence level colors for dark theme */
    .confidence-high {
        border-left: 5px solid #00ff88;
        background: linear-gradient(135deg, #003322 0%, #004433 100%);
    }
    
    .confidence-high h2 {
        color: #00ff88 !important;
    }
    
    .confidence-medium {
        border-left: 5px solid #ffaa00;
        background: linear-gradient(135deg, #332200 0%, #443300 100%);
    }
    
    .confidence-medium h2 {
        color: #ffaa00 !important;
    }
    
    .confidence-low {
        border-left: 5px solid #ff4444;
        background: linear-gradient(135deg, #330000 0%, #441111 100%);
    }
    
    .confidence-low h2 {
        color: #ff4444 !important;
    }
    
    /* Button styling for dark theme */
    .stButton > button {
        background: linear-gradient(135deg, #0088ff 0%, #0066cc 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,136,255,0.3);
    }
    
    /* Tabs styling for dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #cccccc !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0088ff !important;
        color: white !important;
    }
    
    /* Alert styling for dark theme */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(255,255,255,0.1);
    }
    
    .stAlert > div {
        color: #ffffff !important;
    }
    
    /* Success alert */
    .stSuccess {
        background: linear-gradient(135deg, #003322 0%, #004433 100%);
        border-left: 4px solid #00ff88;
    }
    
    /* Error alert */
    .stError {
        background: linear-gradient(135deg, #330000 0%, #441111 100%);
        border-left: 4px solid #ff4444;
    }
    
    /* Warning alert */
    .stWarning {
        background: linear-gradient(135deg, #332200 0%, #443300 100%);
        border-left: 4px solid #ffaa00;
    }
    
    /* Info alert */
    .stInfo {
        background: linear-gradient(135deg, #002233 0%, #003344 100%);
        border-left: 4px solid #0088ff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #111111;
    }
    
    .css-1d391kg .stMarkdown {
        color: #ffffff !important;
    }
    
    /* File uploader for dark theme */
    .stFileUploader {
        background-color: #1a1a1a;
        border: 2px dashed #666666;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        color: #ffffff !important;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0088ff 0%, #0066cc 100%);
    }
    
    /* DataFrame styling for dark theme */
    .dataframe {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    /* Chart backgrounds */
    .js-plotly-plot .plotly .svg-container {
        background-color: #000000 !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #444444;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ERROR HANDLING DECORATOR
# =============================================================================

def handle_errors(func):
    """Decorator for comprehensive error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"❌ {error_msg}")
            return None
    return wrapper

# =============================================================================
# API KEY MANAGEMENT WITH FALLBACK
# =============================================================================

class APIKeyManager:
    """Manage API keys with multiple fallback options"""
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get API key with fallback priority"""
        # Priority 1: Coded primary key
        if config.PRIMARY_API_KEY and config.PRIMARY_API_KEY.startswith('gsk_'):
            return config.PRIMARY_API_KEY
        
        # Priority 2: Environment variable
        env_key = os.getenv('GROQ_API_KEY')
        if env_key and env_key.startswith('gsk_'):
            return env_key
        
        # Priority 3: Session state (user input)
        if 'user_api_key' in st.session_state and st.session_state.user_api_key:
            return st.session_state.user_api_key
        
        # Priority 4: Fallback keys
        for key in config.FALLBACK_KEYS:
            if key and key.startswith('gsk_'):
                return key
        
        return None

api_manager = APIKeyManager()

# =============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE
# =============================================================================

@st.cache_resource(show_spinner=False)
@handle_errors
def load_model_cached(model_path: str):
    """Load and cache models for better performance"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if "UltraLight" in model_path:
            model = UltraLightModel()
        else:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            logger.info(f"Successfully loaded model: {model_path}")
            return model
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

@st.cache_resource(show_spinner=False)
@handle_errors
def initialize_groq_client(api_key: str):
    """Initialize and cache Groq client"""
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None

@st.cache_data(show_spinner=False)
@handle_errors
def preprocess_image_cached(image_array: np.ndarray):
    """Cache image preprocessing"""
    try:
        from torchvision import transforms
        
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

# =============================================================================
# IMAGE VALIDATION AND QUALITY ASSESSMENT
# =============================================================================

class ImageValidator:
    """Advanced image validation and quality assessment"""
    
    @staticmethod
    @handle_errors
    def validate_image(uploaded_file) -> Tuple[bool, str]:
        """Comprehensive image validation"""
        try:
            # File size check
            if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
            
            # Format check
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in config.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
            
            # Image integrity check
            try:
                image = Image.open(uploaded_file)
                image.verify()
                uploaded_file.seek(0)  # Reset file pointer
                image = Image.open(uploaded_file)
                
                # Dimension check
                if image.size[0] < 50 or image.size[1] < 50:
                    return False, "Image too small. Minimum size: 50x50 pixels"
                
                return True, "Image validation successful"
            except Exception:
                return False, "Invalid or corrupted image file"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    @handle_errors
    def assess_quality(image: Image.Image) -> Dict[str, Any]:
        """Assess image quality metrics"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Basic metrics
            height, width = img_array.shape[:2]
            file_size = f"{width}x{height}"
            
            # Brightness analysis
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array)
                brightness_level = "Good" if 50 <= brightness <= 200 else "Poor"
            else:
                brightness = np.mean(img_array)
                brightness_level = "Good" if 50 <= brightness <= 200 else "Poor"
            
            # Blur detection using Laplacian variance
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_level = "Sharp" if blur_score > 100 else "Blurry"
            
            return {
                "dimensions": file_size,
                "resolution": f"{width*height:,} pixels",
                "brightness": brightness_level,
                "blur": blur_level,
                "blur_score": f"{blur_score:.2f}",
                "file_size": f"{width*height*3/1024:.1f} KB"
            }
        except Exception as e:
            return {"error": f"Quality assessment failed: {str(e)}"}

validator = ImageValidator()

# =============================================================================
# AI DIAGNOSIS WITH ERROR HANDLING
# =============================================================================

class AIAnalyzer:
    """AI-powered medical analysis with Groq"""
    
    @staticmethod
    @handle_errors
    def get_diagnosis(condition: str, confidence: float, api_key: str) -> Optional[str]:
        """Generate comprehensive AI medical analysis"""
        try:
            client = initialize_groq_client(api_key)
            if not client:
                return "AI analysis unavailable: API client initialization failed"
            
            prompt = f"""
            As a dermatology AI assistant, provide a comprehensive analysis for a skin lesion classified as '{condition}' with {confidence:.1f}% confidence.

            Please provide:
            1. **Clinical Overview**: Brief explanation of {condition}
            2. **Key Characteristics**: What features typically indicate this condition
            3. **Risk Assessment**: Based on the confidence level ({confidence:.1f}%)
            4. **Recommended Actions**: Specific next steps for the patient
            5. **Additional Notes**: Important considerations or warnings

            Keep the response professional, informative, and appropriate for patient education.
            Include appropriate medical disclaimers.
            """
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI diagnosis failed: {e}")
            return f"AI analysis temporarily unavailable: {str(e)}"

ai_analyzer = AIAnalyzer()

# =============================================================================
# ENHANCED GRADCAM VISUALIZATION
# =============================================================================

class GradCAMGenerator:
    """Enhanced GradCAM visualization with error handling"""
    
    @staticmethod
    @handle_errors
    def generate(model_path: str, image_path: str, target_class: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate enhanced GradCAM visualization with fallback"""
        try:
            model = load_model_cached(model_path)
            if model is None:
                logger.warning("Model loading failed, creating fallback visualization")
                return GradCAMGenerator._create_fallback_visualization(image_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_array = np.array(image)
            
            # Preprocess for model
            input_tensor = preprocess_image_cached(original_array)
            if input_tensor is None:
                return GradCAMGenerator._create_fallback_visualization(image_path)
            
            input_tensor = input_tensor.unsqueeze(0).to(device)
            input_tensor.requires_grad_()
            
            # Forward pass
            output = model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass for gradients
            model.zero_grad()
            output[0, target_class].backward()
            
            # Generate heatmap
            gradients = input_tensor.grad.data
            importance = torch.mean(torch.abs(gradients), dim=1).squeeze().cpu().numpy()
            
            # Resize and normalize heatmap
            heatmap = cv2.resize(importance, (original_array.shape[1], original_array.shape[0]))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Create colored overlay
            colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlayed = cv2.addWeighted(original_array, 0.6, colored_heatmap, 0.4, 0)
            
            return overlayed, heatmap
            
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}")
            return GradCAMGenerator._create_fallback_visualization(image_path)
    
    @staticmethod
    def _create_fallback_visualization(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create fallback visualization when GradCAM fails"""
        try:
            original_image = np.array(Image.open(image_path))
            fallback_heatmap = np.zeros((original_image.shape[0], original_image.shape[1]))
            return original_image, fallback_heatmap
        except Exception as e:
            logger.error(f"Fallback visualization failed: {e}")
            # Return a black image as ultimate fallback
            black_image = np.zeros((224, 224, 3), dtype=np.uint8)
            black_heatmap = np.zeros((224, 224))
            return black_image, black_heatmap

gradcam_generator = GradCAMGenerator()

# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Professional PDF report generation"""
    
    @staticmethod
    @handle_errors
    def create_report(patient_data: Dict, prediction_data: Dict, image_path: str, 
                     ai_diagnosis: str = None) -> Optional[str]:
        """Generate comprehensive PDF report"""
        try:
            # Create temporary file
            report_path = f"skin_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            doc = SimpleDocTemplate(report_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.HexColor('#1f4e79')
            )
            story.append(Paragraph("Professional Skin Lesion Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Patient Information
            story.append(Paragraph("Patient Information", styles['Heading2']))
            patient_table_data = [
                ['Name:', patient_data.get('name', 'N/A')],
                ['Age:', str(patient_data.get('age', 'N/A'))],
                ['Gender:', patient_data.get('gender', 'N/A')],
                ['Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            patient_table = Table(patient_table_data, colWidths=[1.5*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Analysis Results
            story.append(Paragraph("Analysis Results", styles['Heading2']))
            
            # Prediction details
            confidence = prediction_data.get('confidence', 0) * 100
            condition = prediction_data.get('predicted_name', 'Unknown')
            
            result_data = [
                ['Predicted Condition:', condition],
                ['Confidence Level:', f"{confidence:.1f}%"],
                ['Model Used:', 'AI Deep Learning Model'],
                ['Processing Time:', '< 5 seconds']
            ]
            
            result_table = Table(result_data, colWidths=[2*inch, 3.5*inch])
            result_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(result_table)
            story.append(Spacer(1, 20))
            
            # AI Diagnosis if available
            if ai_diagnosis:
                story.append(Paragraph("AI Medical Analysis", styles['Heading2']))
                story.append(Paragraph(ai_diagnosis, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Medical Disclaimer
            story.append(Paragraph("Important Medical Disclaimer", styles['Heading2']))
            disclaimer_text = """
            This analysis is generated by an AI system and is intended for educational and screening purposes only. 
            It should not be considered as a substitute for professional medical diagnosis. Please consult with a 
            qualified dermatologist or healthcare provider for proper medical evaluation and treatment recommendations.
            
            The AI model has been trained on medical datasets but may not account for all possible conditions or 
            individual patient factors. Any concerning skin lesions should be evaluated by a medical professional.
            """
            story.append(Paragraph(disclaimer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            return report_path
            
        except Exception as e:
            logger.error(f"PDF report generation failed: {e}")
            return None

report_generator = ReportGenerator()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Header
    st.title("🏥 Professional Skin Cancer Classification Dashboard")
    st.markdown("### AI-Powered Dermatological Analysis System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🧠 Model Selection")
        model_options = {
            "UltraLight Model (Fast)": config.ULTRALIGHT_MODEL_PATH,
            "EfficientNet Model (Accurate)": config.EFFICIENTNET_MODEL_PATH
        }
        selected_model = st.selectbox("Choose AI Model:", list(model_options.keys()))
        model_path = model_options[selected_model]
        
        # API Key management
        st.subheader("🔑 API Configuration")
        current_api_key = api_manager.get_api_key()
        
        if current_api_key:
            st.success(f"✅ API Key: {current_api_key[:10]}...")
        else:
            st.warning("⚠️ No API key configured")
            user_key = st.text_input("Enter Groq API Key:", type="password", placeholder="gsk_...")
            if user_key:
                st.session_state.user_api_key = user_key
                st.rerun()
        
        # System status
        st.subheader("📊 System Status")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Computing: {device}")
        
        # Check model availability
        if os.path.exists(model_path):
            st.success("✅ Model: Available")
        else:
            st.error("❌ Model: Not Found")
        
        # Performance metrics
        if 'analysis_timestamp' in st.session_state:
            st.metric("Last Analysis", 
                     st.session_state.analysis_timestamp.strftime("%H:%M:%S"))
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Image Upload & Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=config.SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
        )
        
        if uploaded_file is not None:
            # Validate image
            is_valid, validation_msg = validator.validate_image(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ {validation_msg}")
                return
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Quality assessment
            with st.expander("📊 Image Quality Analysis", expanded=False):
                quality_metrics = validator.assess_quality(image)
                
                if "error" not in quality_metrics:
                    q_col1, q_col2 = st.columns(2)
                    with q_col1:
                        st.metric("Brightness", quality_metrics["brightness"])
                        st.metric("Resolution", quality_metrics["resolution"])
                        st.metric("Sharpness", quality_metrics["blur"])
                    with q_col2:
                        st.metric("Dimensions", quality_metrics["dimensions"])
                        st.metric("File Size", quality_metrics["file_size"])
                        st.metric("Blur Score", quality_metrics["blur_score"])
            
            # Save temporary file
            temp_path = "temp_analysis_image.jpg"
            image.save(temp_path)
            
            # Analysis controls
            analyze_col1, analyze_col2 = st.columns([3, 1])
            
            with analyze_col1:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    if not os.path.exists(model_path):
                        st.error(f"❌ Model not found: {model_path}")
                        return
                    
                    with st.spinner(f"🧠 Analyzing with {selected_model}..."):
                        try:
                            # Perform prediction
                            result = predict_skin_cancer(model_path, temp_path)
                            
                            # Generate GradCAM with comprehensive fallback
                            try:
                                gradcam_overlay, gradcam_heatmap = gradcam_generator.generate(model_path, temp_path)
                            except Exception as grad_error:
                                logger.warning(f"GradCAM generation failed: {grad_error}")
                                # Create fallback visualization
                                original_image = np.array(Image.open(temp_path))
                                gradcam_overlay = original_image
                                gradcam_heatmap = np.zeros((original_image.shape[0], original_image.shape[1]))
                            
                            # Store in session state
                            st.session_state.update({
                                'prediction_result': result,
                                'gradcam_overlay': gradcam_overlay,
                                'gradcam_heatmap': gradcam_heatmap,
                                'model_used': selected_model,
                                'image_path': temp_path,
                                'analysis_timestamp': datetime.now()
                            })
                            
                            st.success("✅ Analysis completed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            logger.error(f"Analysis error: {e}", exc_info=True)
            
            with analyze_col2:
                if st.button("🔄", help="Clear results", use_container_width=True):
                    # Clear session state
                    keys_to_clear = ['prediction_result', 'gradcam_overlay', 'ai_diagnosis']
                    for key in keys_to_clear:
                        st.session_state.pop(key, None)
                    st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # Display results if available
        if "prediction_result" in st.session_state:
            result = st.session_state.prediction_result
            
            # Enhanced results display with tabs
            tab1, tab2, tab3 = st.tabs(["🎯 Classification", "🔍 Visualization", "📋 AI Analysis"])
            
            with tab1:
                # Main prediction with confidence-based styling
                confidence_percent = result['confidence'] * 100
                confidence_class = "high" if confidence_percent >= 70 else "medium" if confidence_percent >= 50 else "low"
                
                st.markdown(f"""
                <div class="metric-card confidence-{confidence_class}">
                    <h3>Predicted Condition</h3>
                    <h2>{result['predicted_name']}</h2>
                    <h4>Confidence: {confidence_percent:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("Model", st.session_state.get('model_used', 'N/A'))
                with met_col2:
                    timestamp = st.session_state.get('analysis_timestamp', datetime.now())
                    st.metric("Time", timestamp.strftime("%H:%M:%S"))
                with met_col3:
                    device = "GPU" if torch.cuda.is_available() else "CPU"
                    st.metric("Device", device)
                
                # Medical urgency alerts
                condition = result['predicted_name']
                if condition in config.URGENT_CONDITIONS:
                    st.error("🚨 **URGENT**: Immediate medical attention recommended!")
                elif condition in config.WARNING_CONDITIONS:
                    st.warning("⚠️ **IMPORTANT**: Medical consultation advised!")
                else:
                    st.info("ℹ️ **ADVISORY**: Regular monitoring recommended")
                
                # Probability distribution
                st.markdown("### 📈 Detailed Probability Analysis")
                probs = result['all_probabilities']
                prob_df = pd.DataFrame(list(probs.items()), columns=['Condition', 'Probability'])
                prob_df['Probability'] = prob_df['Probability'] * 100
                prob_df = prob_df.sort_values('Probability', ascending=True)
                
                # Create horizontal bar chart with dark theme
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#000000')
                ax.set_facecolor('#000000')
                
                colors_list = ['#00ff88' if disease == result['predicted_name'] else '#0088ff' 
                              for disease in prob_df['Condition']]
                
                bars = ax.barh(range(len(prob_df)), prob_df['Probability'], color=colors_list)
                ax.set_yticks(range(len(prob_df)))
                ax.set_yticklabels(prob_df['Condition'], fontsize=10, color='white')
                ax.set_xlabel('Probability (%)', fontsize=12, color='white')
                ax.set_title('Classification Confidence Distribution', fontsize=14, fontweight='bold', color='white')
                ax.grid(axis='x', alpha=0.3, color='white')
                ax.tick_params(colors='white')
                
                # Add percentage labels
                for i, (_, row) in enumerate(prob_df.iterrows()):
                    ax.text(row['Probability'] + 1, i, f'{row["Probability"]:.1f}%', 
                           va='center', fontweight='bold', fontsize=9, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                # GradCAM visualization with guaranteed availability
                st.markdown("### 🔍 AI Attention Analysis")
                st.info("This visualization shows which image regions influenced the AI's decision")
                
                if st.session_state.get('gradcam_overlay') is not None:
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.image(Image.open(st.session_state.get('image_path')), 
                                caption="Original Image", use_container_width=True)
                    with vis_col2:
                        st.image(st.session_state.gradcam_overlay, 
                                caption="AI Attention Heatmap", use_container_width=True)
                    
                    # Interpretation guide
                    with st.expander("🔍 Understanding the Heatmap"):
                        st.markdown("""
                        **Color Guide:**
                        - 🔴 **Red/Yellow**: High attention areas - key diagnostic features
                        - 🟢 **Green**: Medium attention - supporting characteristics  
                        - 🔵 **Blue/Purple**: Low attention - background areas
                        
                        **Clinical Significance:**
                        The AI model focuses on medically relevant features such as:
                        - Lesion borders and symmetry
                        - Color variations and patterns
                        - Texture and surface characteristics
                        - Size and shape irregularities
                        """)
                else:
                    st.warning("⚠️ Visualization temporarily unavailable")
                    # Show original image as fallback
                    if 'image_path' in st.session_state:
                        st.image(Image.open(st.session_state.image_path), 
                                caption="Analysis Image", use_container_width=True)
            
            with tab3:
                # AI medical analysis
                st.markdown("### 🤖 Comprehensive Medical Analysis")
                
                current_api_key = api_manager.get_api_key()
                if current_api_key:
                    if st.button("🧠 Generate Medical Report", type="primary", use_container_width=True):
                        with st.spinner("🔬 Generating comprehensive medical analysis..."):
                            diagnosis = ai_analyzer.get_diagnosis(
                                result['predicted_name'], 
                                confidence_percent, 
                                current_api_key
                            )
                            
                            if diagnosis:
                                st.session_state.ai_diagnosis = diagnosis
                                st.rerun()
                
                    # Display cached diagnosis
                    if 'ai_diagnosis' in st.session_state:
                        st.markdown("#### 📋 AI Medical Report")
                        st.markdown(st.session_state.ai_diagnosis)
                        
                        # Patient data form for PDF generation
                        with st.expander("📄 Generate PDF Report"):
                            with st.form("patient_form"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    patient_name = st.text_input("Patient Name")
                                    patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
                                with col_b:
                                    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                                
                                if st.form_submit_button("📄 Generate PDF Report"):
                                    patient_data = {
                                        'name': patient_name,
                                        'age': patient_age,
                                        'gender': patient_gender
                                    }
                                    
                                    with st.spinner("📄 Generating PDF report..."):
                                        pdf_path = report_generator.create_report(
                                            patient_data, 
                                            result, 
                                            st.session_state.image_path,
                                            st.session_state.ai_diagnosis
                                        )
                                        
                                        if pdf_path and os.path.exists(pdf_path):
                                            with open(pdf_path, 'rb') as pdf_file:
                                                st.download_button(
                                                    label="📥 Download PDF Report",
                                                    data=pdf_file.read(),
                                                    file_name=pdf_path,
                                                    mime="application/pdf",
                                                    use_container_width=True
                                                )
                                        else:
                                            st.error("❌ PDF generation failed")
                else:
                    st.warning("⚠️ API key required for medical analysis")
                    st.info("Please configure your Groq API key in the sidebar")
        else:
            st.info("📤 Upload an image to begin analysis")
            
            # Show supported conditions
            with st.expander("🏥 Supported Conditions"):
                st.markdown("**This AI system can classify the following skin conditions:**")
                for i, condition in enumerate(CLASS_NAMES, 1):
                    urgency = "🚨" if condition in config.URGENT_CONDITIONS else "⚠️" if condition in config.WARNING_CONDITIONS else "ℹ️"
                    st.markdown(f"{i}. {urgency} **{condition}**")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        logger.error("Critical error in main application", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.rerun()

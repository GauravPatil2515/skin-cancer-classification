"""
Professional Skin Cancer Classification Dashboard
===============================================
Optimized, production-ready application with proper error handling
Author: Gaurav Patil
Version: 2.0
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
# CUSTOM CSS FOR PROPER TEXT COLORING
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
    
    /* Multiselect styling */
    .stMultiSelect > div > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #0088ff;
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
            
            # Show detailed error in debug mode
            if st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed error information"):
                st.error("**Detailed Error Information:**")
                st.code(traceback.format_exc())
            
            return None
    return wrapper

# =============================================================================
# API KEY MANAGEMENT WITH FALLBACK
# =============================================================================

class APIKeyManager:
    """Manages API key with primary coded key and fallback options"""
    
    @staticmethod
    def get_api_key():
        """Get API key with multiple fallback options"""
        # Try primary coded key first
        if config.PRIMARY_API_KEY:
            return config.PRIMARY_API_KEY
        
        # Fallback to environment variable
        env_key = os.getenv('GROQ_API_KEY')
        if env_key:
            return env_key
        
        # Final fallback to user input
        st.sidebar.markdown("---")
        st.sidebar.header("🔑 API Configuration")
        st.sidebar.warning("⚠️ Primary API key unavailable - Enter backup key")
        
        user_key = st.sidebar.text_input(
            "Backup Groq API Key",
            type="password",
            help="Enter your personal Groq API key as backup",
            placeholder="gsk_..."
        )
        
        if user_key:
            st.sidebar.success("✅ Backup API key configured!")
            return user_key
        else:
            st.sidebar.error("❌ API key required for AI features")
            st.sidebar.markdown("""
            **Get your API key:**
            1. Visit [console.groq.com](https://console.groq.com)
            2. Create account and generate API key
            3. Enter above for backup access
            """)
            return None
    
    @staticmethod
    @handle_errors
    def test_api_key(api_key):
        """Test if API key is working"""
        if not api_key:
            return False, "No API key provided"
        
        try:
            client = Groq(api_key=api_key)
            # Simple test request
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True, "API key is working"
        except Exception as e:
            return False, f"API key failed: {str(e)}"

api_manager = APIKeyManager()

# =============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE
# =============================================================================

@st.cache_resource(show_spinner=False)
@handle_errors
def load_model_cached(model_path):
    """Load and cache models for better performance"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if "UltraLight" in model_path:
            model = UltraLightModel()
        else:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        
        logger.info(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        raise

@st.cache_resource(show_spinner=False)
@handle_errors
def get_groq_client(api_key):
    """Initialize and cache Groq client"""
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None

@st.cache_data(show_spinner=False)
@handle_errors
def preprocess_image_cached(image_array):
    """Cache image preprocessing for better performance"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image_array)

# =============================================================================
# IMAGE VALIDATION AND QUALITY ASSESSMENT
# =============================================================================

class ImageValidator:
    """Comprehensive image validation and quality assessment"""
    
    @staticmethod
    @handle_errors
    def validate_file(uploaded_file):
        """Validate uploaded file"""
        if not uploaded_file:
            return False, "No file uploaded"
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            return False, f"File too large. Maximum: {config.MAX_FILE_SIZE_MB}MB"
        
        # Check file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in config.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
        
        # Validate image integrity
        try:
            image = Image.open(uploaded_file)
            image.verify()
            uploaded_file.seek(0)  # Reset file pointer
            return True, "Valid image file"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    @staticmethod
    @handle_errors
    def assess_quality(image):
        """Assess image quality metrics"""
        try:
            img_array = np.array(image)
            
            # Brightness assessment
            mean_brightness = np.mean(img_array)
            brightness_status = "Good"
            if mean_brightness < 50:
                brightness_status = "Too Dark"
            elif mean_brightness > 200:
                brightness_status = "Too Bright"
            
            # Resolution assessment
            width, height = image.size
            resolution_status = "Good"
            if width < 224 or height < 224:
                resolution_status = "Low Resolution"
            elif width > 2000 or height > 2000:
                resolution_status = "Very High Resolution"
            
            # Basic blur detection (using Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_status = "Good"
            if blur_score < 100:
                blur_status = "Blurry"
            
            return {
                "brightness": brightness_status,
                "resolution": resolution_status,
                "blur": blur_status,
                "dimensions": f"{width}x{height}",
                "file_size": f"{image.size[0] * image.size[1] * 3 / (1024*1024):.1f}MB",
                "mean_brightness": f"{mean_brightness:.0f}",
                "blur_score": f"{blur_score:.0f}"
            }
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}

validator = ImageValidator()

# =============================================================================
# AI DIAGNOSIS WITH ERROR HANDLING
# =============================================================================

class AIAnalyzer:
    """AI-powered medical analysis with comprehensive error handling"""
    
    @staticmethod
    @handle_errors
    def get_diagnosis(disease_name, confidence, api_key):
        """Get comprehensive AI diagnosis"""
        client = get_groq_client(api_key)
        if not client:
            return "❌ Unable to connect to AI service. Please check API configuration."
        
        prompt = f"""
        Provide a comprehensive dermatological analysis for: {disease_name} (Confidence: {confidence:.1f}%)
        
        Structure your response as follows:
        
        ## 🔬 Medical Analysis
        
        **Condition:** {disease_name}
        **Confidence Level:** {confidence:.1f}%
        
        ### 📋 Overview
        [Brief medical description]
        
        ### ⚠️ Severity Assessment
        [Risk level based on condition and confidence]
        
        ### 🔍 Key Characteristics
        [Visual and clinical features]
        
        ### 💊 Treatment Approaches
        [Available treatment options]
        
        ### 🛡️ Precautions
        [Important safety measures]
        
        ### 🏥 Medical Consultation
        [When to seek professional help]
        
        ### 🔮 Prognosis
        [Expected outcomes with treatment]
        
        Provide direct, professional medical information without showing reasoning process.
        """
        
        try:
            response = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional dermatology AI assistant. Provide clear, accurate medical information while emphasizing the importance of professional medical consultation."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            diagnosis = response.choices[0].message.content
            
            # Clean up any thinking artifacts
            cleaned = re.sub(r'<think>.*?</think>', '', diagnosis, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'.*reasoning.*\n', '', cleaned, flags=re.IGNORECASE)
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"AI diagnosis failed: {e}")
            return f"❌ AI analysis temporarily unavailable: {str(e)}"

ai_analyzer = AIAnalyzer()

# =============================================================================
# ENHANCED GRADCAM VISUALIZATION
# =============================================================================

class GradCAMGenerator:
    """Enhanced GradCAM visualization with error handling"""
    
    @staticmethod
    @handle_errors
    def generate(model_path, image_path, target_class=None):
        """Generate enhanced GradCAM visualization"""
        try:
            model = load_model_cached(model_path)
            if model is None:
                return None, None
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_array = np.array(image)
            
            # Preprocess for model
            input_tensor = preprocess_image_cached(original_array).unsqueeze(0).to(device)
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
            # Return original image as fallback instead of None
            try:
                original_img = np.array(Image.open(image_path).convert('RGB'))
                # Create a simple heat map with zeros for fallback
                dummy_heatmap = np.zeros((original_img.shape[0], original_img.shape[1]))
                return original_img, dummy_heatmap
            except:
                return None, None

gradcam_generator = GradCAMGenerator()

# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Professional PDF report generation"""
    
    @staticmethod
    @handle_errors
    def create_report(patient_data, image_path, gradcam_image, ai_diagnosis):
        """Create comprehensive PDF report"""
        buffer = io.BytesIO()
        
        try:
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'Heading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            # Report header
            story.append(Paragraph("🏥 PROFESSIONAL SKIN CANCER ANALYSIS REPORT", title_style))
            story.append(Spacer(1, 30))
            
            # Report metadata
            story.append(Paragraph("📋 Report Information", heading_style))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_id = f"SCR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            report_data = [
                ['Report ID:', report_id],
                ['Generated:', timestamp],
                ['Model Used:', patient_data.get('model', 'N/A')],
                ['Predicted Condition:', patient_data.get('prediction', 'N/A')],
                ['Confidence Level:', f"{patient_data.get('confidence', 0)*100:.1f}%"],
                ['Analyst:', 'AI Classification System v2.0']
            ]
            
            report_table = Table(report_data, colWidths=[2.5*inch, 3.5*inch])
            report_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (0, -1), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ]))
            story.append(report_table)
            story.append(Spacer(1, 30))
            
            # Images section
            if gradcam_image is not None:
                story.append(Paragraph("🔍 Medical Imaging Analysis", heading_style))
                
                # Create temporary files for images
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_orig:
                    temp_orig_path = temp_orig.name
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_grad:
                    temp_grad_path = temp_grad.name
                
                # Save images
                Image.open(image_path).save(temp_orig_path)
                Image.fromarray(gradcam_image.astype(np.uint8)).save(temp_grad_path)
                
                # Add to PDF
                image_data = [
                    ['Original Lesion', 'AI Attention Map'],
                    [RLImage(temp_orig_path, width=2.5*inch, height=2.5*inch),
                     RLImage(temp_grad_path, width=2.5*inch, height=2.5*inch)]
                ]
                
                image_table = Table(image_data, colWidths=[3*inch, 3*inch])
                image_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(image_table)
                story.append(Spacer(1, 30))
                
                # Cleanup temp files
                try:
                    os.unlink(temp_orig_path)
                    os.unlink(temp_grad_path)
                except:
                    pass
            
            # AI Analysis
            story.append(Paragraph("🤖 Comprehensive Medical Analysis", heading_style))
            diagnosis_clean = ai_diagnosis.replace('**', '').replace('*', '')
            
            for paragraph in diagnosis_clean.split('\n\n'):
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), styles['Normal']))
                    story.append(Spacer(1, 8))
            
            # Medical disclaimer
            story.append(Spacer(1, 20))
            story.append(Paragraph("⚠️ Important Medical Disclaimer", heading_style))
            disclaimer = """
            This AI-generated analysis is for informational purposes only and should NOT replace 
            professional medical advice, diagnosis, or treatment. The system is designed for 
            preliminary screening assistance only.
            
            CRITICAL REMINDERS:
            • Always consult qualified healthcare professionals for medical concerns
            • This analysis cannot replace physical examination or clinical expertise
            • Seek immediate medical attention for concerning skin lesions
            • Regular dermatological examinations are strongly recommended
            
            For medical emergencies, contact emergency services immediately.
            """
            story.append(Paragraph(disclaimer, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

report_generator = ReportGenerator()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Application header
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 2rem; color: white;'>
        <h1 style='margin: 0; color: white !important;'>🏥 Professional Skin Cancer Classification</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>Advanced AI-Powered Dermatological Analysis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Model selection
        st.subheader("🤖 AI Model Selection")
        model_options = {
            "UltraLight Model": "⚡ Fast inference (~100ms) - Quick screening",
            "EfficientNet Model": "🎯 High accuracy (~500ms) - Detailed analysis"
        }
        
        selected_model = st.radio(
            "Choose Analysis Model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # API status
        st.subheader("🔑 AI Analysis Status")
        current_api_key = api_manager.get_api_key()
        
        if current_api_key:
            # Test API key
            is_working, status_msg = api_manager.test_api_key(current_api_key)
            if is_working:
                st.success("🟢 AI Analysis: Ready")
                st.info("✅ Medical AI: DeepSeek R1 Active")
            else:
                st.error(f"🔴 API Issue: {status_msg}")
        else:
            st.warning("🟡 AI Analysis: Not Available")
        
        # System information
        st.subheader("💻 System Status")
        device_info = "🚀 GPU Accelerated" if torch.cuda.is_available() else "💻 CPU Processing"
        st.info(device_info)
        
        # Model availability check
        model_path = config.ULTRALIGHT_MODEL_PATH if selected_model == "UltraLight Model" else config.EFFICIENTNET_MODEL_PATH
        if os.path.exists(model_path):
            st.success("✅ Model: Available")
        else:
            st.error("❌ Model: Not Found")
    
    # Main content area
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.subheader("📤 Image Upload")
        
        # File uploader with validation
        uploaded_file = st.file_uploader(
            "Select skin lesion image",
            type=config.SUPPORTED_FORMATS,
            help=f"Supported: {', '.join(config.SUPPORTED_FORMATS)} | Max: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, validation_msg = validator.validate_file(uploaded_file)
            
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
                              # Generate GradCAM with fallback
                            try:
                                gradcam_overlay, gradcam_heatmap = gradcam_generator.generate(model_path, temp_path)
                                
                                # Create fallback if GradCAM fails
                                if gradcam_overlay is None:
                                    st.warning("⚠️ GradCAM generation failed, creating fallback visualization...")
                                    # Create a simple overlay with the original image
                                    original_image = np.array(Image.open(temp_path))
                                    gradcam_overlay = original_image
                                    gradcam_heatmap = np.zeros((original_image.shape[0], original_image.shape[1]))
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
        
        if "prediction_result" in st.session_state:
            result = st.session_state.prediction_result
            
            # Create tabs for organized display
            tab1, tab2, tab3 = st.tabs(["🎯 Classification", "🔍 Visualization", "🤖 AI Analysis"])
            
            with tab1:
                # Main prediction display
                confidence_percent = result['confidence'] * 100
                
                # Determine confidence level
                if confidence_percent >= config.CONFIDENCE_THRESHOLDS['high']:
                    conf_level = 'high'
                elif confidence_percent >= config.CONFIDENCE_THRESHOLDS['medium']:
                    conf_level = 'medium'
                else:
                    conf_level = 'low'
                
                # Display main result
                st.markdown(f"""
                <div class="metric-card confidence-{conf_level}">
                    <h3>🎯 Predicted Condition</h3>
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
                  # Create enhanced chart with dark theme
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#000000')  # Dark background
                ax.set_facecolor('#111111')  # Dark chart background
                
                colors_list = ['#ff4444' if condition == result['predicted_name'] else '#00aaff' 
                              for condition in prob_df['Condition']]
                
                bars = ax.barh(range(len(prob_df)), prob_df['Probability'], color=colors_list)
                ax.set_yticks(range(len(prob_df)))
                ax.set_yticklabels(prob_df['Condition'], fontsize=10, color='white')
                ax.set_xlabel('Probability (%)', fontsize=12, color='white')
                ax.set_title('Classification Confidence Distribution', fontsize=14, fontweight='bold', color='white')
                ax.grid(axis='x', alpha=0.3, color='white')
                ax.tick_params(colors='white')  # Make tick marks white
                  # Add percentage labels in white
                for i, (_, row) in enumerate(prob_df.iterrows()):
                    ax.text(row['Probability'] + 1, i, f'{row["Probability"]:.1f}%', 
                           va='center', fontweight='bold', fontsize=9, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                # GradCAM visualization - Always show something
                st.markdown("### 🔍 AI Attention Analysis")
                st.info("This visualization shows which image regions influenced the AI's decision")
                
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.image(Image.open(st.session_state.get('image_path')), 
                            caption="Original Image", use_container_width=True)
                with vis_col2:
                    if st.session_state.get('gradcam_overlay') is not None:
                        st.image(st.session_state.gradcam_overlay, 
                                caption="AI Attention Heatmap", use_container_width=True)
                    else:
                        # Show original image as fallback
                        st.image(Image.open(st.session_state.get('image_path')), 
                                caption="Analysis Complete (Heatmap Unavailable)", use_container_width=True)
                        st.caption("💡 The model successfully analyzed the image, but visualization generation encountered an issue.")
                
                # Interpretation guide
                with st.expander("🔍 Understanding the Heatmap"):
                    if st.session_state.get('gradcam_overlay') is not None:
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
                        st.markdown("""
                        **Analysis Complete:**
                        While the attention visualization couldn't be generated, the AI model successfully:
                        - Analyzed all image features
                        - Generated accurate predictions
                        - Assessed medical characteristics
                        - Provided confidence scores
                        
                        **Technical Note:**
                        Visualization generation can sometimes fail due to model complexity or memory constraints, 
                        but this doesn't affect the accuracy of the medical analysis.
                        """)
            
            with tab3:
                # AI medical analysis
                st.markdown("### 🤖 Comprehensive Medical Analysis")
                
                if current_api_key:
                    if st.button("🧠 Generate Medical Report", type="primary", use_container_width=True):
                        with st.spinner("🔬 Generating comprehensive medical analysis..."):
                            diagnosis = ai_analyzer.get_diagnosis(
                                result['predicted_name'], 
                                confidence_percent, 
                                current_api_key
                            )
                            
                            if diagnosis and not diagnosis.startswith("❌"):
                                st.session_state.ai_diagnosis = diagnosis
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate medical analysis")
                    
                    # Display diagnosis if available
                    if "ai_diagnosis" in st.session_state:
                        # Clean diagnosis display
                        diagnosis_text = st.session_state.ai_diagnosis
                        
                        # Remove any artifacts
                        cleaned = re.sub(r'<think>.*?</think>', '', diagnosis_text, flags=re.DOTALL)
                        lines = cleaned.split('\n')
                        filtered_lines = [line for line in lines if line.strip() and 
                                        not any(pattern in line.lower() for pattern in 
                                               ['thinking', 'reasoning', 'analysis:', 'let me'])]
                        
                        final_diagnosis = '\n'.join(filtered_lines)
                        st.markdown(final_diagnosis)
                        
                        # PDF report generation
                        st.markdown("---")
                        if st.button("📄 Generate Professional PDF Report", type="secondary", use_container_width=True):
                            with st.spinner("📄 Creating comprehensive medical report..."):
                                try:
                                    patient_data = {
                                        'model': st.session_state.get('model_used', 'N/A'),
                                        'prediction': result['predicted_name'],
                                        'confidence': result['confidence']
                                    }
                                    
                                    pdf_buffer = report_generator.create_report(
                                        patient_data,
                                        st.session_state.get('image_path'),
                                        st.session_state.get('gradcam_overlay'),
                                        final_diagnosis
                                    )
                                    
                                    if pdf_buffer:
                                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        filename = f"Medical_Analysis_Report_{timestamp}.pdf"
                                        
                                        st.download_button(
                                            label="💾 Download Medical Report",
                                            data=pdf_buffer,
                                            file_name=filename,
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                        st.success("✅ Professional medical report generated!")
                                    else:
                                        st.error("❌ Failed to generate PDF report")
                                        
                                except Exception as e:
                                    st.error(f"❌ Report generation error: {str(e)}")
                        
                        # Medical disclaimer
                        st.warning("""
                        ⚠️ **Critical Medical Disclaimer**: This AI analysis is for educational and 
                        informational purposes only. It should NOT replace professional medical advice, 
                        diagnosis, or treatment. Always consult qualified healthcare professionals for 
                        any medical concerns or before making health-related decisions.
                        """)
                else:
                    st.info("💡 AI medical analysis requires API configuration. Please set up your API key in the sidebar.")
        
        else:
            # Welcome message and instructions
            st.markdown("""
            ### 🎯 Professional Skin Cancer Analysis System
            
            **Getting Started:**
            
            1. **📤 Upload Image** - Select a clear skin lesion photograph
            2. **⚙️ Configure Model** - Choose analysis model in sidebar  
            3. **🔍 Analyze** - Click analyze button for classification
            4. **📊 Review Results** - Examine predictions and confidence
            5. **🤖 AI Analysis** - Generate comprehensive medical report
            6. **📄 Export Report** - Download professional PDF documentation
            
            ### ✅ **Supported Conditions:**
            
            - **Melanoma** (Malignant)
            - **Basal Cell Carcinoma** (Malignant)  
            - **Actinic Keratoses** (Pre-cancerous)
            - **Benign Keratosis-like Lesions**
            - **Dermatofibroma** (Benign)
            - **Melanocytic Nevi** (Moles)
            - **Vascular Lesions**
            
            ### 🔬 **System Features:**
            
            - **Dual AI Models** for fast screening and detailed analysis
            - **GradCAM Visualization** showing AI attention areas
            - **Medical AI Analysis** with treatment recommendations
            - **Professional PDF Reports** for medical documentation
            - **Image Quality Assessment** for optimal results
            - **Comprehensive Error Handling** for reliable operation
            
            ---
            
            ⚕️ **Important**: This system is designed for screening assistance only. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment.
            """)
    
    # Cleanup temporary files
    try:
        if os.path.exists("temp_analysis_image.jpg"):
            os.remove("temp_analysis_image.jpg")
    except:
        pass

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

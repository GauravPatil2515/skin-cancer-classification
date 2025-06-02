@echo off
echo 🏥 Professional Skin Cancer Classification Dashboard
echo ================================================
echo.
echo Starting the application...
echo Please wait while dependencies are verified...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Install critical dependencies if needed
echo 📦 Checking dependencies...
pip install streamlit torch pillow opencv-python groq reportlab matplotlib numpy pandas python-dotenv --upgrade --quiet

REM Check if main app exists
if not exist "main_app.py" (
    echo ❌ main_app.py not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Launch the application
echo 🚀 Launching Streamlit application...
echo.
echo Opening in your default browser...
echo To stop the application, press Ctrl+C in this window
echo.

streamlit run main_app.py

pause

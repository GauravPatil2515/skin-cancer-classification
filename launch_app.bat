@echo off
echo ===============================================
echo    Professional Skin Cancer Classification
echo              Dashboard Launcher
echo ===============================================
echo.
echo Starting Streamlit server...
echo.
echo The application will be available at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
streamlit run main_app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false
pause

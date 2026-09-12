@echo off
echo ========================================================
echo   EduPulse AI - Classroom Video Engagement Profiling
echo ========================================================
echo   Starting Streamlit Server...
echo   Open your browser at: http://localhost:8501
echo ========================================================
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
pause

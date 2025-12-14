@echo off
echo Starting Data Center Energy Optimization Dashboard...
echo.
python -m streamlit run dashboard.py --server.headless true
pause

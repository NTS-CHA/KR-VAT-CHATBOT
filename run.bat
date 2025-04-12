@echo off
cd /d "%~dp0vat_chatbot"

echo [1] Creating virtual environment...
if not exist venv (
  python -m venv venv
)

echo [2] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3] Launching FastAPI server...
start "" venv\Scripts\python.exe -m uvicorn main:app --reload

echo [4] Opening Jupyter Notebook for analysis...
start "" venv\Scripts\jupyter-notebook.exe analysis.ipynb

pause

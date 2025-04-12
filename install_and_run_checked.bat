@echo off
cd /d "%~dp0vat_chatbot"

echo == Step 1: Creating virtual environment...
if not exist venv (
  python -m venv venv
)

echo == Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo == Step 3: Installing required Python packages...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install --upgrade "openai>=1.0.0"
venv\Scripts\python.exe -m pip install matplotlib pandas numpy jupyter tiktoken python-dotenv beautifulsoup4 requests fastapi uvicorn pdfkit wordcloud

echo == Step 4: Launching FastAPI server...
start "" "venv\Scripts\python.exe" -m uvicorn main:app --reload

echo == Step 5: Opening Jupyter Notebook...
start "" "venv\Scripts\jupyter-notebook.exe" analysis.ipynb

echo == Step 6: Checking for Node.js and npm...
where npm >nul 2>nul
IF ERRORLEVEL 1 (
  echo ERROR: npm is not installed or not in PATH.
  echo Please install Node.js from: https://nodejs.org/
  pause
  exit /b
)

echo == Step 7: Installing Tailwind CSS and dependencies...
npm install -D tailwindcss postcss autoprefixer

echo == Step 8: Initializing Tailwind config...
npx tailwindcss init -p

echo == Step 9: Building Tailwind CSS (watch mode)...
npx tailwindcss -c tailwind.config.js -i ./static/input.css -o ./static/output.css --watch
npx tailwindcss -c tailwind.config.js -i ./static/input.css -o ./static/output.css --minify
pause

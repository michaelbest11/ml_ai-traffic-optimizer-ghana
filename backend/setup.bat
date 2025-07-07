@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✔️  Setup complete. Activate with: venv\Scripts\activate
pause

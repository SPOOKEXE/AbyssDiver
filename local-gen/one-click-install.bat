@echo off
echo Installing minimum requirements for one-click installer.
pip install --upgrade pip
pip install -r requirements.txt
py installer.py
pause

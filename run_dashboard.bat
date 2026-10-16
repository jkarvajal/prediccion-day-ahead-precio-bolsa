@echo off
title Dashboard - Precio de bolsa de energia
cd /d "%~dp0"

REM Activa el entorno virtual si existe (.venv o venv)
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

echo Iniciando el dashboard en http://localhost:8501 ...
echo Para detenerlo, cierra esta ventana o presiona Ctrl + C.
echo.

streamlit run app\app.py

echo.
echo El dashboard se detuvo. Presiona una tecla para cerrar.
pause >nul

@echo off
chcp 65001 >nul
title upscaler - mac izle
cd /d "C:\Projelerim\upscaler"
rem Optimus: GL baglami NVIDIA'da acilsin (bu laptopta zaten aciliyor, garanti icin)
set SHIM_MCCOMPAT=0x800000001
".venv\Scripts\python.exe" -m upscaler watch %*
if errorlevel 1 (
  echo.
  echo upscaler hata ile kapandi. Kayitlar: C:\Projelerim\upscaler\runs
  pause
)


# VenomAI Pro - Start Backend Server
# ======================================
# Run this script to start the Flask backend and open the app in your browser.
# Usage: Right-click > "Run with PowerShell"  (or run from terminal)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
$backendDir = Join-Path $scriptDir "backend"

if (-Not (Test-Path $venvPython)) {
    Write-Host "❌ Virtual environment not found at: $venvPython" -ForegroundColor Red
    Write-Host "   Please create a venv first: python -m venv .venv" -ForegroundColor Yellow
    Pause
    exit 1
}

Write-Host "🐍 Starting VenomAI Pro Backend..." -ForegroundColor Green
Write-Host "   Server will be available at: http://127.0.0.1:5000" -ForegroundColor Cyan

# Open browser after 2 seconds
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:5000/"
} | Out-Null

# Start Flask server
Set-Location $backendDir
& $venvPython app.py

# setup.ps1
# Cross-platform environment diagnostics (Windows & Linux)
# Does NOT modify PATH — only checks if required tools are usable

Write-Host "Before running this script, make sure to execute:" -ForegroundColor Cyan
Write-Host "  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
Write-Host "Setting up environment..." -ForegroundColor Cyan
Write-Host ""

# ---------------------------------
# Detect operating system
# ---------------------------------
$IsWindowsOS = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows
)

$IsLinuxOS = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Linux
)

if ($IsWindowsOS) {
    Write-Host "Operating system detected: Windows" -ForegroundColor Gray
}
elseif ($IsLinuxOS) {
    Write-Host "Operating system detected: Linux" -ForegroundColor Gray
}
else {
    Write-Host "Operating system detected: Unknown" -ForegroundColor Yellow
}

# ---------------------------------
# 1. Real GNU Make check (Windows & Linux)
# ---------------------------------
function Test-MakeUsable {
    try {
        & make --version | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (Test-MakeUsable) {
    Write-Host "GNU Make found and working (ready to use)." -ForegroundColor Green
} else {
    Write-Warning "GNU Make is not available or not executable."
    Write-Host "-> The 'make' command could not be executed." -ForegroundColor Yellow
    Write-Host "-> Please install GNU Make and ensure it is available in your PATH." -ForegroundColor Yellow
    Write-Host ""

    if ($IsWindowsOS) {
        Write-Host "Windows installation suggestions:" -ForegroundColor Cyan
        Write-Host "  - MSYS2 (recommended): https://www.msys2.org/" -ForegroundColor Cyan
        Write-Host "    - Add to PATH: C:\msys64\usr\bin" -ForegroundColor Gray
        Write-Host "  - Chocolatey:" -ForegroundColor Cyan
        Write-Host "    choco install make" -ForegroundColor Gray
    }
    elseif ($IsLinuxOS) {
        Write-Host "Linux installation suggestions:" -ForegroundColor Cyan
        Write-Host "  - Debian/Ubuntu: sudo apt install make" -ForegroundColor Gray
        Write-Host "  - Arch: sudo pacman -S make" -ForegroundColor Gray
        Write-Host "  - Fedora: sudo dnf install make" -ForegroundColor Gray
    }

    Write-Host ""
}

# ---------------------------------
# 2. Python 3.13+ check (cross-platform)
# ---------------------------------
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.1[3-9]") {
        Write-Host "Python 3.13+ detected: $pyVersion" -ForegroundColor Green
    } else {
        throw "Incompatible Python version"
    }
} catch {
    Write-Error "Python 3.13+ not found. Install it from: https://www.python.org"
    exit 1
}

# ---------------------------------
# 3. Virtual environment activation
# ---------------------------------
if ($IsWindowsOS) {
    $venvPath = ".venv\Scripts\Activate.ps1"
} else {
    $venvPath = ".venv/bin/activate"
}

if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "Virtual environment (.venv) not found." -ForegroundColor Gray
    Write-Host "-> Create it with: python -m venv .venv" -ForegroundColor Gray
}

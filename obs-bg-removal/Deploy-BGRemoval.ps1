# ============================================================
# Deploy-BGRemoval.ps1 (updated for shads-obs-bg-removal rename)
# Deploys built plugin + data to OBS Studio installation
# Run from anywhere; paths auto-detect from this script's location ($PSScriptRoot).
# ============================================================

$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
# Build config: prefer RelWithDebInfo (the preset default), fall back to Release
$BuildDir    = "$ProjectRoot\build_x64\RelWithDebInfo"
if (-not (Test-Path "$BuildDir\shads-obs-bg-removal.dll") -and
    (Test-Path "$ProjectRoot\build_x64\Release\shads-obs-bg-removal.dll")) {
    $BuildDir = "$ProjectRoot\build_x64\Release"
}
$OBSDir      = "C:\Program Files\obs-studio"
$PluginDir   = "$OBSDir\obs-plugins\64bit"
$DataDir     = "$OBSDir\data\obs-plugins\shads-obs-bg-removal"

if (-not (Test-Path $OBSDir)) {
    Write-Host "ERROR: OBS not found at $OBSDir. Set `$OBSDir at the top of this script to your OBS install path." -ForegroundColor Red
    exit 1
}

# Check build exists
$DLL = "$BuildDir\shads-obs-bg-removal.dll"
if (-not (Test-Path $DLL)) {
    # Fall back to old name during transition
    $DLL = "$BuildDir\plugintemplate-for-obs.dll"
    if (-not (Test-Path $DLL)) {
        Write-Host "ERROR: Plugin DLL not found. Run cmake --build first." -ForegroundColor Red
        exit 1
    }
    Write-Host "NOTE: Still using old name 'plugintemplate-for-obs'. Rename in CMakeLists.txt to complete M9." -ForegroundColor Yellow
}

Write-Host "Deploying BG Removal plugin..." -ForegroundColor Cyan

# Plugin DLL
Copy-Item $DLL "$PluginDir\" -Force
Write-Host "  DLL -> $PluginDir" -ForegroundColor Green

# ONNX Runtime (only if not already deployed)
$OrtDLL = "$PluginDir\onnxruntime.dll"
if (-not (Test-Path $OrtDLL)) {
    $SrcOrt = @(
        "$ProjectRoot\deps\onnxruntime\lib\onnxruntime.dll"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($SrcOrt) {
        Copy-Item $SrcOrt $PluginDir\ -Force
        Write-Host "  onnxruntime.dll -> $PluginDir" -ForegroundColor Green
    }
}

$OrtShared = "$PluginDir\onnxruntime_providers_shared.dll"
if (-not (Test-Path $OrtShared)) {
    $SrcShared = "$ProjectRoot\deps\onnxruntime\lib\onnxruntime_providers_shared.dll"
    if (Test-Path $SrcShared) {
        Copy-Item $SrcShared $PluginDir\ -Force
        Write-Host "  onnxruntime_providers_shared.dll -> $PluginDir" -ForegroundColor Green
    }
}

# Data files (shader + model)
New-Item -ItemType Directory -Path "$DataDir\effects" -Force | Out-Null
New-Item -ItemType Directory -Path "$DataDir\models" -Force | Out-Null

Copy-Item "$ProjectRoot\data\effects\mask.effect" "$DataDir\effects\" -Force
Write-Host "  mask.effect -> $DataDir\effects" -ForegroundColor Green

Copy-Item "$ProjectRoot\data\models\rvm_mobilenetv3_fp32.onnx" "$DataDir\models\" -Force
Write-Host "  rvm model -> $DataDir\models" -ForegroundColor Green

# Clean up old plugin name if present
$OldDLL = "$PluginDir\plugintemplate-for-obs.dll"
$OldData = "$OBSDir\data\obs-plugins\plugintemplate-for-obs"
if ((Test-Path $OldDLL) -and (Test-Path $DLL) -and ($DLL -notlike "*plugintemplate*")) {
    Remove-Item $OldDLL -Force
    Write-Host "  Removed old plugintemplate-for-obs.dll" -ForegroundColor DarkYellow
}
if (Test-Path $OldData) {
    Remove-Item $OldData -Recurse -Force
    Write-Host "  Removed old plugintemplate-for-obs data folder" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "Deploy complete! Restart OBS to load the plugin." -ForegroundColor Green

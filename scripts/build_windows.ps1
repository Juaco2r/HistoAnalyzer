$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python scripts/validate_build_layout.py
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt

# All OpenCV wheels install into the same cv2 namespace. Dependencies may pull
# in a second variant, producing an incomplete or inconsistent frozen module.
# Remove every variant, then install exactly one known-good headless wheel.
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>$null
python -m pip install --force-reinstall --no-deps --no-cache-dir opencv-python-headless==4.10.0.84
python scripts/verify_opencv_ml.py
python scripts/make_icons.py

Remove-Item -Recurse -Force dist, build/.pyinstaller -ErrorAction SilentlyContinue
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/.pyinstaller build/HistoAnalyzer.spec

$Exe = Join-Path $Root "dist/HistoAnalyzer/HistoAnalyzer.exe"
$PythonDll = Join-Path $Root "dist/HistoAnalyzer/_internal/python311.dll"
if (-not (Test-Path $Exe)) {
    throw "Windows executable was not created: $Exe"
}
if (-not (Test-Path $PythonDll)) {
    throw "PyInstaller runtime DLL was not created: $PythonDll"
}

# Test the frozen executable, not merely the build environment. The marker is
# written by the GUI executable itself and confirms that ANN_MLP, RTrees and
# FileStorage survived PyInstaller collection.
$SelfTest = Join-Path $Root "build/opencv_ml_frozen_self_test.json"
Remove-Item -Force $SelfTest -ErrorAction SilentlyContinue
$Process = Start-Process -FilePath $Exe -ArgumentList @(
    "--self-test-opencv",
    "--self-test-output",
    $SelfTest
) -Wait -PassThru
if ($Process.ExitCode -ne 0) {
    throw "Frozen OpenCV ML self-test exited with code $($Process.ExitCode)."
}
if (-not (Test-Path $SelfTest)) {
    throw "Frozen OpenCV ML self-test did not create: $SelfTest"
}
$Report = Get-Content -Raw $SelfTest | ConvertFrom-Json
if (-not $Report.ok) {
    throw "Frozen OpenCV ML self-test failed: $($Report.error)"
}
Write-Host "Frozen OpenCV ML self-test passed using $($Report.loader_mode)."

New-Item -ItemType Directory -Force release | Out-Null

# Portable release. The complete folder must be extracted before launching;
# a PyInstaller onedir executable cannot be run by itself from inside a ZIP.
$PortableStage = Join-Path $Root "release/HistoAnalyzer-Windows-x64-PORTABLE"
$PortableZip = Join-Path $Root "release/HistoAnalyzer-Windows-x64-PORTABLE-EXTRACT-FIRST.zip"
Remove-Item -Recurse -Force $PortableStage -ErrorAction SilentlyContinue
Remove-Item -Force $PortableZip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $PortableStage | Out-Null
Copy-Item -Recurse -Force "dist/HistoAnalyzer" (Join-Path $PortableStage "HistoAnalyzer")
Copy-Item -Force "docs/WINDOWS_PORTABLE_README.txt" (Join-Path $PortableStage "README_FIRST.txt")
Compress-Archive -Path "$PortableStage/*" -DestinationPath $PortableZip -CompressionLevel Optimal -Force
Write-Host "Portable build: $PortableZip"

# Installer release (preferred for end users).
$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $iscc = Get-Item $candidate
            break
        }
    }
}
if (-not $iscc) {
    throw "Inno Setup 6 (ISCC.exe) was not found. Install it before building the Windows installer."
}
& $iscc.FullName "build/windows_installer.iss"
$Installer = Join-Path $Root "release/HistoAnalyzer-Windows-x64-Setup.exe"
if (-not (Test-Path $Installer)) {
    throw "Windows installer was not created: $Installer"
}
Write-Host "Installer build: $Installer"

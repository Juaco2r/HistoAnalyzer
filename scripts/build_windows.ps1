$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$Description = "Python command"
    )

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Invoke-PythonChecked -Arguments @("scripts/validate_build_layout.py") -Description "Build layout validation"
Invoke-PythonChecked -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Description "pip upgrade"
Invoke-PythonChecked -Arguments @("-m", "pip", "install", "-r", "requirements-build.txt") -Description "Build dependency installation"

# All OpenCV wheels install into the same cv2 namespace. Dependencies may pull
# in a second variant, producing an incomplete or inconsistent frozen module.
# Query the environment first and uninstall only variants that are actually
# installed. This avoids pip's harmless "Skipping ... not installed" warning,
# which PowerShell can promote to a terminating NativeCommandError when
# $ErrorActionPreference is set to Stop.
$OpenCVVariants = @(
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless"
)

$PipListJson = & python -m pip list --format=json
if ($LASTEXITCODE -ne 0) {
    throw "Could not query installed Python distributions (exit code $LASTEXITCODE)."
}

try {
    $InstalledDistributions = @($PipListJson | ConvertFrom-Json)
}
catch {
    throw "Could not parse 'pip list --format=json': $($_.Exception.Message)"
}

$InstalledNames = @(
    $InstalledDistributions |
        ForEach-Object { [string]$_.name } |
        Where-Object { $OpenCVVariants -contains $_.ToLowerInvariant() }
)

if ($InstalledNames.Count -gt 0) {
    Write-Host "Removing conflicting OpenCV distributions: $($InstalledNames -join ', ')"
    $UninstallArgs = @("-m", "pip", "uninstall", "-y") + $InstalledNames
    Invoke-PythonChecked -Arguments $UninstallArgs -Description "OpenCV cleanup"
}
else {
    Write-Host "No conflicting OpenCV distributions are installed."
}

Invoke-PythonChecked -Arguments @(
    "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir",
    "opencv-python-headless==4.10.0.84"
) -Description "Pinned OpenCV installation"
Invoke-PythonChecked -Arguments @("scripts/verify_opencv_ml.py") -Description "OpenCV ML verification"
Invoke-PythonChecked -Arguments @("scripts/make_icons.py") -Description "Icon generation"

Remove-Item -Recurse -Force dist, build/.pyinstaller -ErrorAction SilentlyContinue
Invoke-PythonChecked -Arguments @(
    "-m", "PyInstaller", "--noconfirm", "--clean",
    "--distpath", "dist", "--workpath", "build/.pyinstaller",
    "build/HistoAnalyzer.spec"
) -Description "PyInstaller build"

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
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}
$Installer = Join-Path $Root "release/HistoAnalyzer-Windows-x64-Setup.exe"
if (-not (Test-Path $Installer)) {
    throw "Windows installer was not created: $Installer"
}
Write-Host "Installer build: $Installer"

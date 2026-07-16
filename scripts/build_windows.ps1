$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python scripts/validate_build_layout.py
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python scripts/make_icons.py
Remove-Item -Recurse -Force dist, build/.pyinstaller -ErrorAction SilentlyContinue
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/.pyinstaller build/HistoAnalyzer.spec
New-Item -ItemType Directory -Force release | Out-Null
Compress-Archive -Path dist/HistoAnalyzer -DestinationPath release/HistoAnalyzer-Windows-x64.zip -Force
Write-Host "Portable build: release/HistoAnalyzer-Windows-x64.zip"
if (Get-Command iscc -ErrorAction SilentlyContinue) {
    iscc build/windows_installer.iss
}

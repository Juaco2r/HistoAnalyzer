from pathlib import Path


def test_windows_installer_resolves_one_executable_path():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert 'Get-Command "iscc.exe" -CommandType Application' in script
    assert "Select-Object -First 1" in script
    assert "$IsccPath" in script
    assert "$InstallerScript" in script
    assert "& $IsccPath $InstallerScript" in script
    assert "$iscc.FullName" not in script


def test_windows_installer_uses_absolute_specification_path():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert 'Resolve-Path -LiteralPath (Join-Path $Root "build/windows_installer.iss")' in script
    assert 'Write-Host "Inno Setup compiler: $IsccPath"' in script
    assert 'Write-Host "Inno Setup script: $InstallerScript"' in script

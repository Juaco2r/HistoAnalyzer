#define MyAppName "HistoAnalyzer"
#define MyAppVersion "1.0.11"
#define MyAppPublisher "José Rodríguez-Rojas"
#define MyAppExeName "HistoAnalyzer.exe"

[Setup]
AppId={{A3A5F572-3F2A-4E47-B00F-907D32A88D83}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\HistoAnalyzer
DefaultGroupName=HistoAnalyzer
OutputDir=..\release
OutputBaseFilename=HistoAnalyzer-Windows-x64-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\HistoAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HistoAnalyzer"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\HistoAnalyzer"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch HistoAnalyzer"; Flags: nowait postinstall skipifsilent

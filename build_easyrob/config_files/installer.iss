[Setup]
AppName=EasyROB
AppVersion=0.1
DefaultDirName={pf}\EasyROB
DefaultGroupName=EasyROB
OutputDir=..\win32_dist\installer
OutputBaseFilename=easyrob_installer
Compression=lzma
SolidCompression=yes
DisableDirPage=no
PrivilegesRequired=admin

[Files]
; Main executable
Source: "..\win32_dist\EasyRob\easyrob.exe"; DestDir: "{app}"; Flags: ignoreversion

; Internal resources
Source: "..\win32_dist\EasyRob\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs ignoreversion

; Conda environment archive
Source: "..\win32_dist\EasyRob\robert_env.tar.gz"; DestDir: "{app}"; Flags: ignoreversion

; Post-installation script
Source: "postinstall.bat"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Shortcut to launch the app
Name: "{group}\EasyROB"; Filename: "{app}\easyrob.exe"; WorkingDir: "{app}"

; Optional: shortcut on desktop
Name: "{commondesktop}\EasyROB"; Filename: "{app}\easyrob.exe"; Tasks: desktopicon; WorkingDir: "{app}"

; Uninstaller
Name: "{group}\Uninstall EasyROB"; Filename: "{uninstallexe}"

[Run]
; Run the postinstall script silently
Filename: "{app}\postinstall.bat"; Flags: runhidden waituntilterminated skipifdoesntexist

[Tasks]
; Optional desktop icon task
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

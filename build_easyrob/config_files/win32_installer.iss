[Setup]
AppName=easyROB
AppVersion=0.5
DefaultDirName={commonpf}\easyROB
DefaultGroupName=easyROB
OutputDir=..\distribution\
OutputBaseFilename=easyrob_installer
Compression=lzma
SolidCompression=yes
DisableDirPage=no
PrivilegesRequired=admin

[Files]
; Main executable
Source: "..\distribution\build\easyrob.exe"; DestDir: "{app}"; Flags: ignoreversion

; Internal resources
Source: "..\distribution\build\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs ignoreversion

; Post-installation script
Source: "postinstall.bat"; DestDir: "{app}"; Flags: ignoreversion

; Postuninstall script
Source: "postuninstall_cleanup.bat"; DestDir: "{app}"; Flags: ignoreversion


[Icons]
; Shortcut to launch the app
Name: "{group}\easyROB"; Filename: "{app}\easyrob.exe"; WorkingDir: "{app}"

; Optional: shortcut on desktop
Name: "{commondesktop}\easyROB"; Filename: "{app}\easyrob.exe"; Tasks: desktopicon; WorkingDir: "{app}"

; Uninstaller
Name: "{group}\Uninstall easyROB"; Filename: "{uninstallexe}"

[Run]
; Run the postinstall script silently
Filename: "{app}\postinstall.bat"; StatusMsg: "Setting up easyROB environment. This may take several minutes..."; Flags: runhidden waituntilterminated skipifdoesntexist

[UninstallRun]
; Run cleanup script silently after uninstall
Filename: "{app}\postuninstall_cleanup.bat"; Flags: runhidden waituntilterminated;

[Tasks]
; Optional desktop icon task
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

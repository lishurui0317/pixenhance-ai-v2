; ============================================================================
; Yupianxiangsu v1.0.0 安装包脚本
; 编译命令: C:\InnoSetup\ISCC.exe build_installer.iss
; ============================================================================

[Setup]
AppId={{B9F4A7E2-3C81-4D6F-8E5A-1A2B3C4D5E6F}
AppName=Yupianxiangsu
AppVersion=1.0.0
AppPublisher=Yupianxiangsu Studio
AppPublisherURL=https://yupianxiangsu.local
AppSupportURL=https://yupianxiangsu.local
AppUpdatesURL=https://yupianxiangsu.local
DefaultDirName={autopf}\Yupianxiangsu
DefaultGroupName=Yupianxiangsu
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=Yupianxiangsu_v1.0.0_Setup
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\Yupianxiangsu.exe
UninstallDisplayName=Yupianxiangsu v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
VersionInfoVersion=1.0.0.0
VersionInfoCompany=Yupianxiangsu Studio
VersionInfoDescription=Yupianxiangsu v1.0.0 Setup
VersionInfoProductName=Yupianxiangsu
DisableWelcomePage=no
DisableFinishedPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
BeveledLabel=Yupianxiangsu - AI Image Upscaler

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional tasks:"
Name: "startmenu"; Description: "Create a Start menu shortcut"; GroupDescription: "Additional tasks:"

[Files]
Source: "dist\Yupianxiangsu.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Yupianxiangsu"; Filename: "{app}\Yupianxiangsu.exe"; IconFilename: "{app}\Yupianxiangsu.exe"; Tasks: startmenu
Name: "{group}\Uninstall Yupianxiangsu"; Filename: "{uninstallexe}"; Tasks: startmenu
Name: "{commondesktop}\Yupianxiangsu"; Filename: "{app}\Yupianxiangsu.exe"; IconFilename: "{app}\Yupianxiangsu.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Yupianxiangsu.exe"; Description: "Launch Yupianxiangsu"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\config.json"
Type: filesandordirs; Name: "{app}\__pycache__"
; 打包版配置存在 %LOCALAPPDATA%\Yupianxiangsu（Program Files 不可写）
Type: filesandordirs; Name: "{localappdata}\Yupianxiangsu"

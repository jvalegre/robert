from pathlib import Path
import sys
import platform

block_cipher = None

# --- PROJECT PATHS ----------------------------------------------------------

SPEC_DIR = Path.cwd().resolve()
print(SPEC_DIR)
PROJECT_ROOT = SPEC_DIR.parent
print(PROJECT_ROOT)
GUI_DIR = PROJECT_ROOT/"GUI_easyROB"
DIST_DIR = SPEC_DIR/"distribution"  
print(PROJECT_ROOT)

# Path for main gui script config
ENTRY_POINT = GUI_DIR/"easyrob.py"               # entry‑point script
APP_NAME = "easyROB"

if sys.platform == 'win32':
    ICON_FILE = str(SPEC_DIR/"config_files"/"Robert_icon.ico")
elif sys.platform == 'darwin':
    ICON_FILE = str(SPEC_DIR/"config_files"/"macOS_icon.icns")
#elif sys.platform == 'linux':
#    ICON_FILE = str(PROJECT_ROOT/"robert"/"report"/"Robert_icon.png")

# --- PYINSTALLER SPECS ----------------------------------------------------------

a = Analysis(
    [ENTRY_POINT],
    pathex=[GUI_DIR],    
    binaries=[],    
    datas=[
        # Include the unpacked environment instead of the tar.gz
        (str(DIST_DIR/'robert_env_unpacked'), 'robert_env'),
        # Include platform-specific files
        *([('postinstall.sh', 'Contents/Resources')] if sys.platform == 'darwin' else [])
    ],
    hiddenimports=[],   
    hookspath=[str(SPEC_DIR)],
    # Add post-installation hook for macOS
    runtime_hooks=['post_install.py'] if sys.platform == 'darwin' else [],
    hooksconfig={},
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure)

# Base executable configuration
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Force one-dir mode on all platforms
    name=APP_NAME,
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=ICON_FILE,
)

# Platform-specific settings
if sys.platform == 'darwin':  # macOS
    # Create .app file for macOS
    app = BUNDLE(
        exe,
        name=f'{APP_NAME}.app',
        icon=ICON_FILE,
        bundle_identifier='com.robert.easyrob',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleShortVersionString': '0.5.0',  # Match Windows version
            'CFBundleVersion': '0.5.0',
            'LSMinimumSystemVersion': '10.13.0',
            # Permissions needed for installation
            'NSAppleEventsUsageDescription': 'Application needs access to file system for environment setup',
            # Environment variables for post-installation
            'LSEnvironment': {
                'INSTALL_SCRIPT': 'Contents/Resources/postinstall.sh'
            }
        },
    )

# Common file collection for all platforms
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="build",
)
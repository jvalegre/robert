from pathlib import Path

block_cipher = None

# --- PROJECT PATHS ----------------------------------------------------------

SPEC_DIR = Path.cwd().resolve()
PROJECT_ROOT = SPEC_DIR.parent
GUI_DIR = PROJECT_ROOT/"GUI_easyROB"
print(PROJECT_ROOT)

# Path for main gui script config
ENTRY_POINT = GUI_DIR/"easyrob.py"               # entry‑point script
ICON_FILE = str(PROJECT_ROOT/"robert"/"report"/"Robert_icon.ico")    # application icon
APP_NAME = "EasyRob"

excluded_modules = [
    'tkinter',
    'unittest',
    'test',
    'distutils',
    'lib2to3',
    'turtle',
    'doctest',
    'idlelib',
    'audioop',
    'ossaudiodev',
    'pydoc',
    'pydoc_data',
    'optparse',
    'cgi',
    'xmlrpc',
    'wave',
    'chunk',
]

# --- PYINSTALLER SPECS ----------------------------------------------------------

a = Analysis(
    [ENTRY_POINT],
    pathex=[GUI_DIR],
    binaries=[],
    datas=[],
    hiddenimports=[],   
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name=APP_NAME,
    console=True,          # True if cmd window needed
    debug=True,             # True if debug mode needed(needs console=True)
    icon=ICON_FILE,         # Path to icon.
    exclude_binaries=False,  
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,          # create dist/<APP_NAME>/
    distpath=str(SPEC_DIR/"win32_dist"),
    workpath=str(SPEC_DIR/"tmp"),
)
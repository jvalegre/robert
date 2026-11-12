import os
import sys
import subprocess
from pathlib import Path

def run_postinstall():
    """
    Execute platform-specific post-installation steps.
    For macOS, this includes:
    - Locating the postinstall script in the app bundle
    - Setting execution permissions
    - Running the script to unpack the conda environment
    """
    if sys.platform == 'darwin':
        if getattr(sys, 'frozen', False):
            # Get the app bundle path when running as a frozen application
            app_path = Path(sys._MEIPASS)
            install_script = app_path / 'Contents' / 'Resources' / 'postinstall.sh'
            
            if install_script.exists():
                try:
                    # Set execute permissions
                    os.chmod(install_script, 0o755)
                    # Run the postinstall script
                    subprocess.run([str(install_script)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Post-installation error: {e}")

if __name__ == '__main__':
    run_postinstall()

"""
Version utilities for easyROB and its dependencies.

This module collects version information from both Python packages and
external command-line tools used by the application.

Responsibilities:
- Retrieve installed Python package versions
- Query external CLI tools (e.g., OpenBabel, xtb)
- Normalize version outputs for display in the GUI

Notes:
- External tools are queried via subprocess calls
- Failures are handled gracefully by returning "Not found"
- Designed for diagnostic and reproducibility purposes

"""

import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version


EASYROB_VERSION = "2.0.0"

def get_python_package_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "Not found"

def get_cli_version(cmd):
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
        return result.stdout.splitlines()[0]
    except Exception:
        return "Not found"

def get_xtb_version():
    try:
        result = subprocess.run(["xtb", "--version"], capture_output=True, text=True)
        output = result.stdout + result.stderr
        match = re.search(r"xtb version ([0-9.]+)", output)
        if match:
            return match.group(1)
        return "Unknown"
    except Exception:
        return "Not found"

def get_software_versions():
    return {
        "easyROB": EASYROB_VERSION,
        "Dependencies": {
            "Python": platform.python_version(),
            "OpenBabel": get_cli_version("obabel"),
            "xtb": get_xtb_version(),
            "AQME": get_python_package_version("aqme"),
            "ROBERT": get_python_package_version("robert"),
        },
    }

SOFTWARE_VERSIONS = get_software_versions()
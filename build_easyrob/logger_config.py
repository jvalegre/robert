import logging
from pathlib import Path
import sys


def setup_logger():
    main_logger = logging.getLogger("easyrob_build")
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler("build.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)

    # PyInstaller logger
    pyinstaller_logger = logging.getLogger("pyinstaller_logger")
    pyinstaller_logger.setLevel(logging.INFO)
    pyinstaller_logger.propagate = False 

    pyinstaller_log_path = Path("pyinstaller.log")
    pyinstaller_file_handler = logging.FileHandler(pyinstaller_log_path, mode="w")
    pyinstaller_file_handler.setFormatter(logging.Formatter("%(message)s"))
    pyinstaller_logger.addHandler(pyinstaller_file_handler)

    # Conda-pack logger
    conda_pack_logger = logging.getLogger("condapack_logger")
    conda_pack_logger.setLevel(logging.INFO)
    conda_pack_logger.propagate = False

    conda_pack_logger_log_path = Path("conda_pack.log")
    conda_pack_logger_file_handler = logging.FileHandler(
        conda_pack_logger_log_path, mode="w"
    )
    conda_pack_logger_file_handler.setFormatter(logging.Formatter("%(message)s"))
    conda_pack_logger.addHandler(conda_pack_logger_file_handler)

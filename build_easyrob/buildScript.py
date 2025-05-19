import logging
import os
import shutil
import subprocess
from pathlib import Path
import sys
from logger_config import setup_logger

setup_logger()

log = logging.getLogger("easyrob_build")
pyInstaller_log = logging.getLogger("pyinstaller_logger")


def define_env(env_name: str, yaml_path: Path):
    try:
        log.info("Environment check started")
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )
        if any(
            line.startswith(env_name) or f"{env_name} " in line
            for line in result.stdout.splitlines()
        ):
            log.warning(f"Conda environment '{env_name}' already exists.")
            response = (
                input("Do you want to continue using it? [Y/N]: ").strip().lower()
            )
            if response != "y":
                log.info("Aborted by user.")
                sys.exit(0)
        else:
            log.info(f"Creating new environment: {env_name}")
            subprocess.run(
                ["conda", "env", "create", "-n", env_name, "-f", str(yaml_path)],
                check=True,
            )
            log.info(f"Environment '{env_name}' created successfully.")
    except FileNotFoundError as fe:
        log.error(f"Conda is not installed or not in PATH.\n{fe}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to execute conda command:\n{e}")
        sys.exit(1)


def pyinstaller_build(env_name: str, spec_path: Path, dist_dir: Path, work_dir: Path):
    log.info(f"Building executable with PyInstaller and environment '{env_name}'...")
    log.info("The full build log will be saved to 'pyinstaller.log'.")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "-u",
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        str(spec_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        with process.stdout:
            for line in process.stdout:
                pyInstaller_log.info(line.rstrip())

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        log.info("Build completed successfully.")
        log.info("Please, check 'pyinstaller.log' for details and Warnings")
    except subprocess.CalledProcessError as e:
        log.error("Build failed. Check 'pyinstaller.log' for details.")
        log.error(f"Command: {e.cmd}")
        log.error(f"Return code: {e.returncode}")
        sys.exit(1)


def buildContext(root: Path):
    match sys.platform:
        case "win32":
            dist = root / "win32_dist"
            tmp = root / "tmp"
        case "linux":
            dist = root / "linux_dist"
            tmp = root / "tmp"
        case "darwin":
            dist = root / "macos_dist"
            tmp = root / "tmp"
        case _:
            log.error(f"Unsupported platform: {sys.platform}")
            raise RuntimeError("Unsupported platform")
    dist.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return dist, tmp


def main():
    env_name = "easyrob_env"
    rootBuild_dir = Path(__file__).parent
    yaml_path = rootBuild_dir / "easyrob_env.yaml"
    spec_path = rootBuild_dir / "pyinstallerBuild.spec"
    dist_path, tmp_path = buildContext(rootBuild_dir)

    if not spec_path.exists():
        log.error(f"Spec file not found at {spec_path}")
        sys.exit(1)
    if not yaml_path.exists():
        log.error(f"YAML file not found at {yaml_path}")
        sys.exit(1)

    define_env(env_name=env_name, yaml_path=yaml_path)
    pyinstaller_build(
        env_name=env_name, spec_path=spec_path, dist_dir=dist_path, work_dir=tmp_path
    )

    try:
        shutil.rmtree(tmp_path)
        log.info(f"Cleaned temporary build folder: {tmp_path}")
    except Exception as e:
        log.warning(f"Could not remove tmp folder {tmp_path}: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"Unexpected error occurred: {e}")
        sys.exit(1)

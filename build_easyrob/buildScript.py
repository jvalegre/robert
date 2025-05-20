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
conda_pack_log = logging.getLogger("condapack_logger")


def deactivate_env():
    try:
        subprocess.run(
            ["conda", "deactivate"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to execute conda command:\n{e}")
        sys.exit(1)


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
            log.info(response)
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


def conda_pack(env_name: str, output_dir: Path):
    log.info(f"Packing Conda environment '{env_name}' with conda-pack...")
    log.info("The full packing log will be saved to 'conda-pack.log'.")

    archive_name = f"{env_name}.tar.gz"
    output_path = output_dir / archive_name

    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "conda-pack",
        "-n",
        env_name,
        "-o",
        str(output_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with process.stdout:
            for line in process.stdout:
                conda_pack_log.info(line.rstrip())

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        log.info("Environment packed successfully.")
        log.info("You can find the archive at: %s", output_dir)
    except subprocess.CalledProcessError as e:
        log.error("Conda-pack failed. Check 'conda-pack.log' for details.")
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


def build_innosetup_installer(iss_path: Path):
    log.info("Building installer with Inno Setup...")
    log.info("The full log will be shown in console output.")

    iscc_path = Path("C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe")

    if not iscc_path.exists():
        log.error(f"Inno Setup compiler not found at {iscc_path}")
        sys.exit(1)

    command = [str(iscc_path), str(iss_path)]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.info("Installer built successfully.")
        log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error("Inno Setup failed.")
        log.error(f"Command: {e.cmd}")
        log.error(e.stdout)
        sys.exit(1)


def main():
    rootBuild_dir = Path(__file__).parent
    easyrob_env = "easyrob_env"
    easyrob_yaml = rootBuild_dir / "config_files" / "easyrob_env.yaml"
    robert_env = "robert_env"
    robert_yaml = rootBuild_dir / "config_files" / "robert-aqme_env.yaml"
    spec_path = rootBuild_dir / "config_files" / "pyinstallerBuild.spec"
    dist_path, tmp_path = buildContext(rootBuild_dir)
    result_path = dist_path / "EasyRob"
    iss_file = rootBuild_dir / "config_files" / "installer.iss"

    if not spec_path.exists():
        log.error(f"Spec file not found at {spec_path}")
        sys.exit(1)
    if not easyrob_yaml.exists():
        log.error(f"YAML file not found at {easyrob_yaml}")
        sys.exit(1)

    define_env(env_name=easyrob_env, yaml_path=easyrob_yaml)
    pyinstaller_build(
        env_name=easyrob_env, spec_path=spec_path, dist_dir=dist_path, work_dir=tmp_path
    )

    define_env(env_name=robert_env, yaml_path=robert_yaml)
    conda_pack(env_name=robert_env, output_dir=result_path)

    build_innosetup_installer(iss_path=iss_file)

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

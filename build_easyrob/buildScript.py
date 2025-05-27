import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Tuple
from logger_config import setup_logger

setup_logger()

log = logging.getLogger("easyrob_build")


def define_env(env_name: str, yaml_path: Path, force_recreate: bool = False) -> bool:
    """Define or verify a conda environment.
    
    Args:
        env_name: Name of the conda environment
        yaml_path: Path to the YAML file with environment definition
        force_recreate: If True, recreate the environment even if it exists
        
    Returns:
        bool: True if the environment is ready to use, False if there was an error
    
    Raises:
        FileNotFoundError: If conda or YAML file is not found
        subprocess.CalledProcessError: If a conda command fails
    """
    if not yaml_path.exists():
        log.error(f"Configuration file {yaml_path} does not exist")
        return False
        
    try:
        log.info("Checking environment...")
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )
        
        env_exists = any(
            line.startswith(env_name) or f"{env_name} " in line
            for line in result.stdout.splitlines()
        )
        if env_exists:
            if force_recreate:
                log.info(f"Removing existing environment '{env_name}'...")
                subprocess.run(["conda", "env", "remove", "-n", env_name], check=True)
            else:
                log.warning(f"Environment '{env_name}' already exists.")
                while True:
                    response = input("Do you want to continue using it? [Y/N]: ").strip().lower()
                    log.info(f"User response: {response}")
                    match response:
                        case "y":
                            return True
                        case "n":
                            while True:
                                response = input("Do you want delete it and re-create? [Y/N]: ").strip().lower()
                                log.info(f"User response: {response}")
                                match response:
                                    case "y":
                                        log.info(f"Removing existing environment '{env_name}'...")
                                        subprocess.run(["conda", "env", "remove", "-n", env_name], check=True)
                                        break
                                    case "n":
                                        log.info("Process aborted by user.")
                                        return False
                                    case _:
                                        print("Invalid input")
                            break
                        case _:
                            print("Invalid input")
        log.info(f"Creating new environment: {env_name}")
        subprocess.run(
            ["conda", "env", "create", "-n", env_name, "-f", str(yaml_path)],
            check=True,
            capture_output=True,
            text=True
        )
        log.info(f"Environment '{env_name}' created successfully.")
        return True
        
    except FileNotFoundError as fe:
        log.error(f"Conda is not installed or not in PATH.\n{fe}")
        return False
    except subprocess.CalledProcessError as e:
        log.error(f"Error executing conda command:\n{e.stdout}\n{e.stderr}")
        return False


def pyinstaller_build(
    env_name: str, 
    spec_path: Path, 
    dist_dir: Path, 
    work_dir: Path,
    clean: bool = True,
    debug: bool = False,
    one_file: bool = False,
    windowed: bool = False
) -> bool:
    """Build the executable using PyInstaller.
    
    Args:
        env_name: Name of the conda environment to use
        spec_path: Path to the .spec file
        dist_dir: Directory where the executable will be generated
        work_dir: Temporary working directory
        clean: If True, clean temporary files before building
        debug: If True, include debugging information
        one_file: If True, generate a single executable file
        windowed: If True, hide console on Windows
        
    Returns:
        bool: True if build was successful, False if there was an error
    """
    if not spec_path.exists():
        log.error(f"Spec file {spec_path} does not exist")
        return False
        
    log.info(f"Building executable with PyInstaller in environment '{env_name}'...")

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
    ]
    
    if clean:
        command.append("--clean")
    if debug:
        command.append("--debug")
    if one_file:
        command.append("--onefile")
    if windowed:
        command.append("--windowed")
        
    command.extend([
        "--noconfirm",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        str(spec_path),
    ])

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
                log.info(line.rstrip())

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        log.info("Build completed successfully.")
        log.info("Please check 'pyinstaller.log' for details and warnings")
        return True
        
    except subprocess.CalledProcessError as e:
        log.error("Build failed. Check 'pyinstaller.log' for more details.")
        log.error(f"Command: {e.cmd}")
        log.error(f"Return code: {e.returncode}")
        return False


def conda_pack(env_name: str, output_dir: Path):
    log.info(f"Packing Conda environment '{env_name}' with conda-pack...")
    
    archive_name = f"{env_name}.tar.gz"
    output_path = output_dir / archive_name
    unpacked_dir = output_dir / "robert_env_unpacked"

    # Asegúrate de que el directorio destino esté limpio
    if unpacked_dir.exists():
        shutil.rmtree(unpacked_dir)
    unpacked_dir.mkdir(parents=True)

    try:
        # Crear el archivo tar.gz
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

        process = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Descomprimir el archivo
        log.info("Descomprimimiendo el entorno empaquetado...")
        subprocess.run(
            ["tar", "-xzf", str(output_path), "-C", str(unpacked_dir)],
            check=True
        )

        # Ejecutar conda-unpack en el entorno descomprimido
        log.info("Ejecutando conda-unpack...")
        if sys.platform == "win32":
            unpack_script = unpacked_dir / "Scripts" / "conda-unpack.exe"
        else:
            unpack_script = unpacked_dir / "bin" / "conda-unpack"
        
        subprocess.run([str(unpack_script)], check=True, cwd=str(unpacked_dir))

        # Eliminar el archivo tar.gz original ya que no lo necesitaremos
        output_path.unlink()
        
        log.info("Environment unpacked successfully.")
        
    except subprocess.CalledProcessError as e:
        log.error("Failed during environment packing or unpacking.")
        log.error(f"Command: {e.cmd}")
        log.error(f"Return code: {e.returncode}")
        sys.exit(1)


# Default paths used across the build process
def get_build_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """Get standard paths for building.
    
    Returns:
        tuple containing:
        - dist_dir: Distribution directory
        - tmp_dir: Temporary working directory
        - spec_file: PyInstaller spec file
        - installer_config: Platform specific installer config
    """
    dist_dir = root / "distribution"
    tmp_dir = root / "tmp"
    spec_file = root / "config_files" / "pyinstallerBuild.spec"
    
    # Only Windows needs an installer config file
    installer_config = root / "config_files" / "win32_installer.iss" if sys.platform == "win32" else None
    
    return dist_dir, tmp_dir, spec_file, installer_config


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


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build tool for EasyRob')
    
    # Frontend (EasyRob GUI) environment arguments
    frontend_group = parser.add_argument_group('Frontend Environment')
    frontend_group.add_argument('--frontend-env', default='easyrob_env',
                    help='Name of the frontend conda environment (default: easyrob_env)')
    frontend_group.add_argument('--frontend-yaml', type=Path,
                    help='Path to the frontend environment YAML file')
    
    # Backend (Robert) environment arguments
    backend_group = parser.add_argument_group('Backend Environment')
    backend_group.add_argument('--backend-env', default='robert_env',
                    help='Name of the backend conda environment (default: robert_env)')
    backend_group.add_argument('--backend-yaml', type=Path,
                    help='Path to the backend environment YAML file')
    
    # Environment management arguments
    env_group = parser.add_argument_group('Environment Management')
    env_group.add_argument('--force-recreate', action='store_true',
                    help='Force recreation of environments even if they exist')
    env_group.add_argument('--skip-env', action='store_true',
                    help='Skip environment creation/verification')
    
    # Build arguments
    build_group = parser.add_argument_group('Build Options')
    build_group.add_argument('--debug', action='store_true',
                    help='Enable debug mode in PyInstaller')
    build_group.add_argument('--one-file', action='store_true',
                    help='Generate a single executable file')
    build_group.add_argument('--windowed', action='store_true',
                    help='Hide console on Windows')
    build_group.add_argument('--skip-installer', action='store_true',
                    help='Skip installer creation')
    
    # macOS specific arguments
    if sys.platform == "darwin":
        parser.add_argument(
            "--version",
            default="0.5.0",
            help="Version number for macOS package"
        )
        
    return parser.parse_args()

def main():
    """Main build script function."""
    try:
        args = parse_args()
        root = Path(__file__).parent
        
        # Get standard build paths
        dist_dir, tmp_dir, spec_file, installer_config = get_build_paths(root)
        
        # Create required directories
        dist_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Define environments
        if not define_env(args.frontend_env, root/"config_files"/"frontend_env.yaml", args.force_recreate):
            return 1
            
        if not define_env(args.backend_env, root/"config_files"/"backend_env.yaml", args.force_recreate):
            return 1
            
        # Pack conda environment
        conda_pack(args.backend_env, dist_dir)
        
        # Build with PyInstaller
        if not pyinstaller_build(
            args.frontend_env,
            spec_file,
            dist_dir,
            tmp_dir,
            clean=True,
            windowed=args.windowed
        ):
            return 1
            
        # Create platform-specific installer
        if not args.skip_installer:
            if sys.platform == 'win32':
                if not build_innosetup_installer(installer_config):
                    log.error("Error creating Windows installer")
                    return 1
            elif sys.platform == 'darwin':
                if not build_macos_package(dist_dir, "easyROB", args.version):
                    log.error("Error creating macOS package")
                    return 1
                    
        return 0
        
    except Exception as e:
        log.exception(f"Build failed: {e}")
        return 1

def build_macos_package(dist_dir: Path, app_name: str, version: str = "0.5.0"):
    """Build a macOS .pkg installer
    
    Args:
        dist_dir: Directory containing the .app bundle
        app_name: Name of the application
        version: Version string for the package
        
    Returns:
        bool: True if build was successful, False if there was an error
    """
    log.info("Building macOS package...")
    
    try:
        app_path = dist_dir / "build" / f"{app_name}.app"
        pkg_dir = dist_dir / "installer"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the package
        subprocess.run([
            "pkgbuild",
            "--root", str(app_path),
            "--install-location", f"/Applications/{app_name}.app",
            "--identifier", f"com.robert.{app_name.lower()}",
            "--version", version,
            str(pkg_dir / f"{app_name.lower()}_installer.pkg")
        ], check=True)
        
        log.info("macOS package built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        log.error(f"Error building macOS package: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())

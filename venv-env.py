import venv
import subprocess
import sys
from pathlib import Path
import platform
import os
import shutil

def check_python_version():
    required_version = "3.11"
    current_version = platform.python_version()
    return current_version.startswith(required_version)

def find_python_executable():
    python_executables = ["python3.11", "python3", "python"]
    for executable in python_executables:
        if shutil.which(executable):
            return executable
    return None

def install_homebrew():
    print("Homebrew is not installed. Installing Homebrew...")
    homebrew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    try:
        subprocess.run(homebrew_install_cmd, shell=True, check=True)
        print("Homebrew has been successfully installed.")
        # Add Homebrew to PATH for the current session
        os.environ["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{os.environ['PATH']}"
    except subprocess.CalledProcessError:
        print("Failed to install Homebrew. Please install it manually and try again.")
        sys.exit(1)

def install_python():
    required_version = "3.11"
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        # Check if Python 3.11 is already installed
        python_executable = find_python_executable()
        if python_executable:
            result = subprocess.run([python_executable, "--version"], capture_output=True, text=True)
            if result.stdout.startswith(f"Python {required_version}"):
                print(f"Python {required_version} is already installed.")
                return python_executable
        
        # Check if Homebrew is installed
        if not shutil.which("brew"):
            install_homebrew()
        
        # Install Python using Homebrew
        try:
            subprocess.run(["brew", "install", f"python@{required_version}"], check=True)
            print(f"Python {required_version} has been installed successfully.")
            return f"python{required_version}"
        except subprocess.CalledProcessError:
            print(f"Failed to install Python {required_version}. Please install it manually and try again.")
            sys.exit(1)
    elif system == "linux":
        # This assumes a Debian-based system. Adjust for other distributions.
        subprocess.run(["sudo", "apt", "update"])
        subprocess.run(["sudo", "apt", "install", "-y", "software-properties-common"])
        subprocess.run(["sudo", "add-apt-repository", "-y", "ppa:deadsnakes/ppa"])
        subprocess.run(["sudo", "apt", "update"])
        subprocess.run(["sudo", "apt", "install", "-y", f"python{required_version}"])
        return f"python{required_version}"
    else:
        print(f"Unsupported operating system: {system}")
        sys.exit(1)

def create_venv():
    python_executable = find_python_executable()
    if not python_executable or not check_python_version():
        print(f"Python 3.11 is not the current Python version. Attempting to install/use it...")
        python_executable = install_python()
    
    venv_path = Path(".venv")
    
    # Create virtual environment
    subprocess.run([python_executable, "-m", "venv", str(venv_path)])

    # Determine the path to the Python executable in the virtual environment
    if sys.platform == "win32":
        python_executable = venv_path / "Scripts" / "python.exe"
        activate_script = venv_path / "Scripts" / "activate"
    else:
        python_executable = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"

    # Upgrade pip
    subprocess.run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])

    # Update requirements.txt with current package versions
    update_requirements()

    print(f"Virtual environment created at {venv_path}")

    # Activate the virtual environment
    if sys.platform == "win32":
        activate_command = str(activate_script)
    else:
        activate_command = f"source {activate_script}"

    # Use a shell to run the activation command and then install requirements
    shell_command = f"{activate_command} && {str(python_executable)} -m pip install -r requirements.txt"
    subprocess.run(shell_command, shell=True)

    print("Virtual environment has been activated and requirements have been installed.")

def update_requirements():
    # Read existing requirements
    try:
        with open("requirements.txt", "r") as f:
            existing_requirements = f.read().splitlines()
    except FileNotFoundError:
        existing_requirements = []

    # Get installed packages
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True
    )

    # Filter out packages that are typically not needed in requirements.txt
    packages_to_exclude = {'pip', 'setuptools', 'wheel', 'distribute'}
    installed_packages = [
        line for line in result.stdout.split('\n')
        if line and not any(package in line for package in packages_to_exclude)
    ]

    # Merge existing requirements with installed packages
    updated_requirements = list(set(existing_requirements + installed_packages))
    updated_requirements.sort()

    # Ensure Python 3.11 is specified
    python_requirement = "python==3.11.*"
    if python_requirement not in updated_requirements:
        updated_requirements.insert(0, python_requirement)

    with open("requirements.txt", "w") as f:
        f.write("\n".join(updated_requirements))

    print("requirements.txt has been updated with current package versions.")

if __name__ == "__main__":
    create_venv()
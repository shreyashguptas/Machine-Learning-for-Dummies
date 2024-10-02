import venv
import subprocess
import sys
from pathlib import Path

def create_venv():
    venv_path = Path(".venv")
    venv.create(venv_path, with_pip=True)

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

    # Install requirements
    subprocess.run([str(python_executable), "-m", "pip", "install", "-r", "requirements.txt"])

    print(f"Virtual environment created at {venv_path}")
    print(f"To activate the environment:")
    if sys.platform == "win32":
        print(f"    {activate_script}")
    else:
        print(f"    source {activate_script}")

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

    with open("requirements.txt", "w") as f:
        f.write("\n".join(updated_requirements))

    print("requirements.txt has been updated with current package versions.")

if __name__ == "__main__":
    create_venv()
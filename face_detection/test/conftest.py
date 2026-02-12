import os
import sys
import subprocess


def _venv_site_packages(venv_path: str) -> str:
    """Get the site-packages directory from the venv"""
    py = os.path.join(venv_path, "bin", "python")
    try:
        return subprocess.check_output(
            [py, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True
        ).strip()
    except Exception:
        return None


# Setup PYTHONPATH to include AI venv before importing anything else
VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")
site_pkgs = _venv_site_packages(VENV_PATH)

if site_pkgs and os.path.exists(site_pkgs):
    if site_pkgs not in sys.path:
        sys.path.insert(0, site_pkgs)
    print(f"[pytest] Using AI venv site-packages: {site_pkgs}")
else:
    print(f"[pytest] Warning: Could not find venv at {VENV_PATH}")

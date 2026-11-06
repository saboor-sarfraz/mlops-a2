#!/usr/bin/env python3
"""
Task 1.1 - Environment report.

Prints the interpreter, the OS, the exact version of every library used in the
project, and whether TensorFlow can see a GPU. Also writes the full list of
installed packages to requirements_frozen.txt (the same information as
`pip freeze`, produced without shelling out).

Run from S:\\MLOps\\A2 with the venv active:
    python env_report.py
"""
import os
import platform
import sys
from datetime import datetime
from importlib import metadata

# Keep TensorFlow start-up noise out of the screenshot.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Libraries the project depends on (distribution names as pip knows them).
KEY_PACKAGES = [
    "tensorflow", "keras", "numpy", "pillow", "matplotlib", "pandas",
    "scikit-learn", "scipy", "keras-tuner", "mlflow", "dagshub", "dvc",
    "flask", "waitress", "python-dotenv", "sqlalchemy",
]


def version_of(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main():
    print("=" * 64)
    print("VISIONGUARD - ENVIRONMENT REPORT")
    print("=" * 64)
    print(f"Date/time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"User             : {os.environ.get('USERNAME') or os.environ.get('USER')}")
    print(f"Python version   : {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Operating system : {platform.platform()}")
    print(f"Virtual env      : {os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX') or sys.prefix}")

    print("\nKey library versions")
    print("-" * 64)
    for pkg in KEY_PACKAGES:
        print(f"{pkg:<18} {version_of(pkg)}")

    print("\nGPU check")
    print("-" * 64)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} built with CUDA: "
          f"{tf.sysconfig.get_build_info().get('is_cuda_build')}")
    if gpus:
        for gpu in gpus:
            print("GPU found:", gpu)
    else:
        print("No GPU detected by TensorFlow - training will run on the CPU.")
    print("CPU devices:", [d.name for d in tf.config.list_physical_devices("CPU")])

    # Full package list for reproducibility (also used in Task 3 as an artifact).
    dists = sorted(
        ((d.metadata["Name"], d.version) for d in metadata.distributions()),
        key=lambda item: item[0].lower(),
    )
    with open("requirements_frozen.txt", "w", encoding="utf-8") as f:
        for name, ver in dists:
            f.write(f"{name}=={ver}\n")
    print(f"\nWrote requirements_frozen.txt ({len(dists)} packages)")


if __name__ == "__main__":
    main()

"""Sets up the project."""

import pathlib

from setuptools import setup


repo_root = pathlib.Path(__file__).absolute().parent
package_name = "xcolumns"


def get_version():
    """Gets the package version from __init__.py"""
    try:
        path = repo_root / package_name / "__init__.py"
        content = path.read_text(encoding="utf-8")

        for line in content.splitlines():
            if line.startswith("__version__"):
                return line.strip().split()[-1].strip().strip('"')

    except Exception:
        raise RuntimeError("Failed to retrieve the package version.")


def get_long_description():
    """Gets the long description from the README.md"""
    try:
        path = repo_root / "README.md"
        return path.read_text(encoding="utf-8")

    except Exception:
        raise RuntimeError("Failed to retrieve the package description.")


setup(
    name=package_name,
    version=get_version(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
)

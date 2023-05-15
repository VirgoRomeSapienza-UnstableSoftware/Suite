"""
Setuptools based setup module

this is used to upload to PyPi
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    # Name of the project, registered the first time this was uploaded
    name="virgoSuite",  #   Required
    setuptools_git_versioning={
        "enabled": True,
        "template": "{tag}",
    },
    setup_requires=["setuptools-git-versioning"],
    description="Toolbox used from the data analysis group of Virgo Rome",  # Optional
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Riccardo Felicetti",
    author_email="riccardo.felicetti@infn.it",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "xarray",
        "pandas",
        "astropy",
        "zarr",
        "tqdm",
        "requests",
    ],
)

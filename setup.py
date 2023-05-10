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
    version_config=True,
    version_config={
        "template": "{tag}",
    },
    setup_requires=["setuptools-git-versioning"],
    version="0.0.2",  #   Required
    description="Toolbox used from the data analysis group of Virgo Rome",  # Optional
    long_description=long_description,
    author="Riccardo Felicetti",
    author_email="riccardo.felicetti@infn.it",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
    ],
    packages=find_packages(include=["virgoSuie", "virgoSuite.*"]),
)

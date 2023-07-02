# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys, os

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "virgoSuite"
copyright = "2023, Virgo-Rome"
author = "Felicetti Riccardo"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "numpydoc",
]

autosummary_generate = ["suite"]
autodoc_mock_imports = ["numpy", "typing", "os", "fnmatch"]


numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- Options for Cross-reference (links) -------------------------------------------------
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

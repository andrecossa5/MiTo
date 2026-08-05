"""Sphinx configuration for the MiTo documentation."""

import os
import sys
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "MiTo"
author = "Andrea Cossa"
copyright = f"2025, {author}"

# Single-source the version from the installed package rather than hard-coding it.
try:
    release = _version("scmito")
except Exception:  # noqa: BLE001  -- docs must build from a bare checkout too
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"       # keep signatures readable
autodoc_docstring_signature = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,             # hide the undocumented internals
    "imported-members": False,          # do not document re-exported third-party functions
    "private-members": False,
    "special-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = True
napoleon_use_rtype = True

# -- Cross-project links -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "never"        # notebooks are committed pre-executed
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
nbsphinx_assume_equations = True

# -- HTML output -------------------------------------------------------------

html_theme = "renku"
html_title = f"MiTo {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_show_sourcelink = True
html_copy_source = False

html_theme_options = {
    # renku is a skin over sphinx-rtd-theme, so it takes the RTD options
    "collapse_navigation": False,
    "navigation_depth": 3,
    "sticky_navigation": True,
    "titles_only": False,
    "logo_only": True,
    "style_external_links": True,
}

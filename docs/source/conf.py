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

html_theme = "furo"
html_title = f"MiTo {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_show_sourcelink = True
html_copy_source = False

html_theme_options = {
    "sidebar_hide_name": True,          # the logo already carries the name
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/andrecossa5/MiTo/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#b3261e",
        "color-brand-content": "#b3261e",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'SF Mono', 'JetBrains Mono', Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8a80",
        "color-brand-content": "#ff8a80",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/andrecossa5/MiTo",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38'
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
                "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66"
                ".07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
                "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27"
                "c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15"
                "0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2"
                '0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

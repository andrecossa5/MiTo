# Configuration file for the Sphinx documentation builder.
# Updated for MiTo v0.1.1 - Enhanced clonal inference with edge case handling

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'MiTo'
author = 'Andrea Cossa'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx.ext.mathjax'
]

autodoc_default_options = {
    'members': True,
    'imported-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
}

# Napoleon configuration for a NumPy style docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = True

# Disable the duplicated signature in the docstring
autodoc_docstring_signature = False

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build (use pre-executed notebooks)
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
nbsphinx_assume_equations = True  # Enable equation support

# -- Options for HTML output -------------------------------------------------
exclude_patterns = ['_build', '**.ipynb_checkpoints']
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
# templates_path = ['_templates']

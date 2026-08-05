Installation
============

MiTo is distributed on PyPI as ``scmito``, and imported as ``mito``.

Quick install
-------------

.. code-block:: bash

   pip install scmito

That is all that is required: every dependency, including the phylogenetics
back-end, is resolved automatically.

Requirements
------------

* **Python 3.11, 3.12 or 3.13**
* A C/C++ toolchain

The compiler is needed because ``cassiopeia-mt`` — MiTo's phylogenetics
dependency — is published as a source distribution and builds a handful of
Cython and C++17 extensions during installation. On macOS the Xcode command line
tools (``xcode-select --install``) are enough; most Linux distributions already
provide ``gcc``. Expect the install to take a few minutes the first time.

In a fresh environment
----------------------

Working in an isolated environment is recommended:

.. code-block:: bash

   # with conda / mamba
   mamba create -n mito python=3.12 -y
   mamba activate mito
   pip install scmito

.. code-block:: bash

   # or with venv
   python -m venv .venv
   source .venv/bin/activate
   pip install scmito

Verify the installation
-----------------------

.. code-block:: python

   import mito as mt

   print(mt.__version__)

Run it from a directory other than a clone of the MiTo repository: a local
``mito`` source folder shadows the installed package.

Development install
-------------------

To work on MiTo itself:

.. code-block:: bash

   git clone https://github.com/andrecossa5/MiTo.git
   cd MiTo
   pip install -e ".[test]"
   pytest

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Contents
   * - ``[test]``
     - ``pytest`` and ``pytest-cov``, for running the test-suite
   * - ``[docs]``
     - Sphinx and the theme used to build this documentation

.. code-block:: bash

   pip install "scmito[test]"

Troubleshooting
---------------

**The build fails while compiling cassiopeia-mt.**
A C++17-capable compiler is missing. Install the Xcode command line tools on
macOS (``xcode-select --install``), or ``build-essential`` on Debian/Ubuntu.

**pip cannot find a compatible version.**
Check your Python version: MiTo requires 3.11 or newer.

**Importing mito fails after a successful install.**
You are most likely inside a clone of the MiTo repository, where the local
``src/mito`` folder shadows the installed package. Change directory and retry.

**Anything else.**
Please open an issue at
`github.com/andrecossa5/MiTo/issues <https://github.com/andrecossa5/MiTo/issues>`_.

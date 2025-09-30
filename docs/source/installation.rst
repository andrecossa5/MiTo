Installation
============

This guide will walk you through installing MiTo, including setting up the necessary environment.

Step 1: Install mamba (or conda)
---------------------------------

First, install mamba for fast and reliable package management:

Install `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ (or conda)

Step 2: Clone MiTo repository
-----------------------------

Clone the MiTo repository from GitHub:

.. code-block:: bash

   git clone https://github.com/andrecossa5/MiTo.git

Step 3: Reproduce MiTo conda environment
----------------------------------------

Navigate to the MiTo directory and create the conda environment:

.. code-block:: bash

   cd MiTo
   mamba env create -f envs/environment.yml -n MiTo

Step 4: Activate environment and install MiTo
----------------------------------------------

Activate the environment and install MiTo via PyPI:

.. code-block:: bash

   mamba activate MiTo
   pip install mito_utils

Step 5: Verify installation
---------------------------

To verify a successful installation, import MiTo in Python:

.. code-block:: python

   import mito as mt

Additional Notes
----------------

For further assistance, please refer to our documentation or open an issue on GitHub.

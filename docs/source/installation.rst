Installation
============

This guide will walk you through installing MiTo, including setting up the necessary environment.

Step 1: Clone MiTo repository
-----------------------------

First, clone the MiTo repository from GitHub and navigate into the project folder:

.. code-block:: bash

   git clone https://github.com/andrecossa5/MiTo.git
   cd MiTo

Step 2: Reproduce MiTo environment
--------------------------------------

MiTo relies on a pre-configured Conda environment to manage dependencies. We recommend using the mamba package manager for a fast and reliable setup. In the ``envs`` folder, the ``environment.yml`` recipe lists all necessary dependencies.

To reproduce this environment, run:

.. code-block:: bash

   mamba env create -f envs/environment.yml -n MiTo
   mamba activate MiTo

Step 3: Manually install cassiopeia
--------------------------------------

We need a specific version of cassiopeia: commit e7606afd10035a75f718ffb988666264e721700e. We will install it with --no-deps flag, 
as all dependencies have been already installed.

.. code-block:: bash

   pip install --no-deps git+https://github.com/YosefLab/Cassiopeia.git@e7606afd10035a75f718ffb988666264e721700e


Step 4: Install MiTo
--------------------

With the environment set up and the manual dependency installed, we can now install MiTo from the project root:

.. code-block:: bash

   pip install .

Step 4: Verify installation
-------------------------------

To verify a successful installation, open a Python interpreter and check the version:

.. code-block:: python

   import mito
   print(mito.__version__)

Additional Notes
----------------

The next release will update cassiopeia dependency, ensuring a more flexible installation.
Please, follow the instruction as indicated and tested.
For further assistance, please refer to our documentation or open an issue on GitHub.

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

MiTo relies on a pre-configured Conda environment to manage dependencies. We recommend using the mamba package manager for a fast and reliable setup. In the ``envs`` folder, you'll find the ``environment.yml`` recipe listing all necessary dependencies.

To reproduce this environment, run:

.. code-block:: bash

   mamba env create -f envs/environment.yml -n MiTo
   mamba activate MiTo

Step 3: Install cassiopeia-lineage
--------------------------------------------

MiTo depends on the cassiopeia-lineage package, which must be installed manually. Run the following command in the activated Conda environment:

.. code-block:: bash

   pip install --no-deps git+https://github.com/YosefLab/Cassiopeia.git@e7606afd10035a75f718ffb988666264e721700e

Step 4: Install MiTo
--------------------

With the environment set up and the manual dependency installed, you can now install MiTo from the project root:

.. code-block:: bash

   pip install .

Step 5: Verify installation
-------------------------------

To verify a successful installation, open a Python interpreter and check the version:

.. code-block:: python

   import mito
   print(mito.__version__)

Additional Notes
----------------

- **Manual Dependency:**  
  MiTo is built upon functionalities provided by cassiopeia-lineage. To ensure seamless compatibility, the specific commit 
  ``e7606afd10035a75f718ffb988666264e721700e`` must be installed. Automatic dependency resolution by pip is disabled for this 
  package, as it might reinstall or upgrade packages already present in the environment, leading to incompatibilities. Please 
  follow the installation step exactly. Future releases of MiTo may address this dependency handling.

- **Troubleshooting:**  
  If you experience issues during installation, please verify that:
  
  - Your Conda environment is active.
  - The manual installation command for cassiopeia-lineage executed without errors.

For further assistance, please refer to our documentation or open an issue on GitHub.

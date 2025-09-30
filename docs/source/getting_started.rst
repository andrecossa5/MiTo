Getting Started
===============

This tutorial gives a complete overview of MiTo's core functionalities for **single-cell mitochondrial lineage tracing**. The interested user is also encouraged to take a look at MiTo's companion pipeline, `nf-MiTo <https://github.com/andrecossa5/nf-MiTo>`_, which automates this data workflow at scale.

.. note::
   A properly configured MiTo environment is a necessary prerequisite for running this tutorial. Follow :doc:`installation` to install MiTo and its dependencies on your machine.

The following interactive notebook provides a comprehensive walkthrough of MiTo's functionality, including all code examples with their outputs and visualizations.

This tutorial showcases the basics of MiTo APIs, which not only backs up nf-MiTo main processes, but also enable interactive exploration of MT-scLT data (after raw sequencing data pre-processing, see `nf-MiTo <https://github.com/andrecossa5/nf-MiTo>`_ documentation). In particular, we will see how to:

1. Load an AFM (AnnData format, output from nf-MiTo PREPROCESS)
2. Pre-process an AFM (i.e., cell and variant filtering, genotyping)
3. Compute pairwise cell-cell distances in the selected MT-SNV space
4. Infer mitochondrial cell phylogenies
5. Infer mitochondrial clones

We will also provide basic examples on how to use MiTo's powerful plotting library.

.. toctree::
   :maxdepth: 2

   getting_started_tutorial

Additional Notes
----------------

- **Getting Help:**  
  If you run into any issues or have suggestions for improvement, please open an issue on our `GitHub repository <https://github.com/andrecossa5/MiTo/issues>`_.

Enjoy using MiTo for your mitochondrial analyses!
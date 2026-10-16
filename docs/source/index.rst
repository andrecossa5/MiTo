MiTo
====

**Mitochondrial lineage tracing and single-cell multi-omics in Python**

MiTo turns mitochondrial DNA variation into single-cell phylogenies. It reads
MAESTER (maegatk), mtscATAC (mgatk) and ReDeeM (RedeemV) output into an
:class:`~anndata.AnnData` Allele Frequency Matrix of molecule counts, selects
informative MT-SNVs, calls single-cell genotypes, reconstructs the cell phylogeny, and
annotates its clones with the support of their clades - with the same calls and
defaults for the three assays.

.. code-block:: bash

   pip install scmito

.. note::

   This is the documentation of **MiTo 2.x**. The original API (1.0.0), used for the analyses of the
   MiTo paper, stays available: ``pip install "scmito<2"`` and its
   `documentation <https://mito.readthedocs.io/en/v1.0.0/>`_. See :doc:`migrating` to move between them.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      The pipeline in four calls, on simulated clones.

   .. grid-item-card:: Loading data
      :link: loading_data
      :link-type: doc

      The AFM, and how to build it from maegatk, mgatk, RedeemV or counts tables.

   .. grid-item-card:: Benchmark
      :link: benchmark
      :link-type: doc

      Simulated data, MAESTER, and ReDeeM vs mtscATAC on the same cells.

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Set up MiTo and its dependencies.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Every public function, organised by module.

   .. grid-item-card:: About
      :link: about
      :link-type: doc

      Background, citation and related work.

Overview
--------

The package follows the scverse layout, so it composes with ``scanpy`` and
``anndata``:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Module
     - Purpose
   * - ``mt.io``
     - Build AFMs from pipeline output, read and write Newick trees
   * - ``mt.pp``
     - Cell and variant filtering, genotype calling, distances, kNN graphs, embeddings
   * - ``mt.tl``
     - Tree building, clade support, clonal inference, clustering, fate bias
   * - ``mt.pl``
     - Trees, heatmaps, embeddings, coverage and variant-spectrum plots
   * - ``mt.ut``
     - Metrics (kBET, ARI, NMI, CI/RI), MT annotations, helpers

At a glance
-----------

.. code-block:: python

   import scanpy as sc
   import mito as mt

   afm = sc.read('afm_unfiltered.h5ad')

   mt.pp.filter_cells(afm, cell_filter='filter2')
   mt.pp.filter_afm(afm)

   tree = mt.tl.build_tree(afm, solver='UPMGA')
   mt.tl.annotate_clones(tree, afm)

   mt.pl.plot_tree(tree, annot='MiTo_clone')

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   installation
   getting_started
   loading_data
   benchmark
   migrating
   biological_example

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   api
   about

Index
-----

* :ref:`genindex`
* :ref:`search`

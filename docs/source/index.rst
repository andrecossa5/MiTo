MiTo
====

**Mitochondrial lineage tracing and single-cell multi-omics in Python**

MiTo turns mitochondrial DNA variation into single-cell phylogenies. It reads
MAESTER, RedeeM, Cas9 and scWGS output into an :class:`~anndata.AnnData` Allele
Frequency Matrix, selects informative MT-SNVs, calls single-cell genotypes,
computes cell–cell distances in mutational space, and reconstructs and annotates
clonal trees.

.. code-block:: bash

   pip install scmito

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Set up MiTo and its dependencies.

   .. grid-item-card:: Tutorial
      :link: getting_started
      :link-type: doc

      A full walkthrough, from raw AFM to an annotated phylogeny.

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
     - Tree building, clonal inference, clustering, fate bias, bootstrapping
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

   afm = mt.pp.filter_cells(afm, cell_filter='filter2')
   afm = mt.pp.filter_afm(afm, filtering='MiTo')

   tree = mt.tl.build_tree(afm, precomputed=True, solver='UPMGA')
   annotator = mt.tl.MiToTreeAnnotator(tree)
   annotator.clonal_inference()

   mt.pl.plot_tree(tree, features=['MiTo clone'])

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   installation
   getting_started

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

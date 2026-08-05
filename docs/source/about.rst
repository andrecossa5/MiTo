About
=====

MiTo is a Python package for **mitochondrial single-cell lineage tracing**
(MT-scLT): it adds the lineage modality encoded in naturally occurring
mitochondrial DNA variants (MT-SNVs) to single-cell workflows.

What it does
------------

1. **Assembles Allele Frequency Matrices** from MT-scLT assays — MAESTER,
   ReDeeM, scmtATAC-seq — into :class:`~anndata.AnnData` objects, with
   alternative-allele and depth counts kept as layers.
2. **Selects informative MT-SNVs** with a choice of strategies
   (``MiTo``, ``MQuad``, ``weng2024``, ``miller2022``, ``CV``, ``baseline``,
   ``GT_enriched``).
3. **Calls single-cell genotypes**, either by hard thresholding (``vanilla``)
   or from binomial-mixture posteriors (``MiTo``).
4. **Reconstructs phylogenies** in mtDNA mutational space, using
   distance-based solvers (``UPMGA``, ``NJ``, ``spectral``, ``greedy``).
5. **Annotates trees** with a bespoke algorithm that resolves discrete clonal
   populations, and quantifies agreement against ground truth where available.

Design
------

MiTo follows the `scverse <https://scverse.org/>`_ conventions — an
``io`` / ``pp`` / ``tl`` / ``pl`` / ``ut`` module layout over
:class:`~anndata.AnnData` — so it composes directly with
`scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and the rest of the
ecosystem. That interoperability is what allows additional modalities (gene
expression, chromatin accessibility, protein abundance) to be mapped onto
inferred phylogenies and clones.

Phylogenetic reconstruction is built on
`Cassiopeia <https://github.com/YosefLab/Cassiopeia>`_, redistributed as
``cassiopeia-mt`` so that MiTo can pin a released version.

At scale
--------

MiTo's core functionality is also packaged as
`nf-MiTo <https://github.com/andrecossa5/nf-MiTo>`_, a Nextflow pipeline for
running MT-SNV-based lineage tracing across many samples.

Citation
--------

If MiTo is useful in your work, please cite:

  Cossa, A. *et al.* **MiTo: mitochondrial lineage tracing and single-cell
  multi-omics.** bioRxiv (2025).
  `doi:10.1101/2025.06.17.660165 <https://doi.org/10.1101/2025.06.17.660165>`_

Please also cite the underlying methods you rely on — in particular
`Cassiopeia <https://doi.org/10.1186/s13059-020-02000-8>`_ for tree
reconstruction, and
`MAESTER <https://doi.org/10.1038/s41587-022-01210-8>`_ for the assay.

License
-------

MIT. See the
`LICENSE <https://github.com/andrecossa5/MiTo/blob/master/LICENSE>`_ file.

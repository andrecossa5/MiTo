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
2. **Selects informative MT-SNVs**: read-level candidate filters, then a
   variant QC on the cell kNN graph that keeps only the MT-SNVs whose carriers
   are clustered, or mutually exclusive with the other variants' carriers.
3. **Calls single-cell genotypes** by testing each cell's alternative reads
   against the variant's own, iteratively estimated, sequencing-error rate.
4. **Reconstructs phylogenies** in mtDNA mutational space, using
   distance-based solvers (``UPMGA``, ``NJ``, ``spectral``, ``greedy``).
5. **Annotates trees** by cutting them where the characters pay for the split,
   abstaining on cells whose calls their neighbourhood does not corroborate, and
   quantifies agreement against ground truth where available.

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
  multi-omics.** Nature Communications (2026).
  `doi:10.1038/s41467-026-71607-5 <https://doi.org/10.1038/s41467-026-71607-5>`_

Please also cite the underlying methods you rely on — in particular
`Cassiopeia <https://doi.org/10.1186/s13059-020-02000-8>`_ for tree
reconstruction, and
`MAESTER <https://doi.org/10.1038/s41587-022-01210-8>`_ for the assay.

License
-------

MIT. See the
`LICENSE <https://github.com/andrecossa5/MiTo/blob/master/LICENSE>`_ file.

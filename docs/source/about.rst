MiTo
==========

MiTo is a user-friendly Python package enabling interactive, single-cell, lineage-informed, multi-omic analysis.

MiTo's capabilities include:

1. Preprocessing of scLT data (particular focus on `MAESTER <10.1038/s41587-022-01210-8>` technology)
2. Phylogeny and clonal inference
3. Downstream analysis and visualization

Implemented in accordance with the `scverse <https://scverse.org/>`_ guidelines and 
integrating with other popular single-cell packages (e.g., `anndata <https://anndata.readthedocs.io/en/stable/>`_,
`scanpy <https://scanpy.readthedocs.io/en/stable/>`_, `cassiopeia <https://cassiopeia-lineage.readthedocs.io/en/latest/index.html>`_),
MiTo adds the "lineage" modality encoded in naturally occurring mitochondrial DNA (mtDNA) 
variants (MT-SNVs) to python single-cell workflows.

MiTo core functionalities have been implemented into `nf-MiTo <https://github.com/andrecossa5/nf-MiTo>`_, a Nextflow pipeline for 
scalable MT-SNVs-based single-cell lineage tracing.

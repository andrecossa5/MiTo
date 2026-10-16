How MiTo works
==============

One pipeline serves the three assays: MAESTER, ReDeeM and mtscATAC. What differs is how the
counts are read into the AFM (:doc:`loading_data`); from there on every stage, default and
threshold is the same.

.. code-block:: python

   mt.pp.filter_cells(afm, cell_filter='filter2')   # cells
   mt.pp.filter_afm(afm)                            # variants, genotypes, distances
   tree = mt.tl.build_tree(afm, solver='UPMGA')     # phylogeny
   mt.tl.annotate_clones(tree, afm)                 # clones, with clade support

All functions work in place on the :class:`~anndata.AnnData` (pass ``copy=True`` for a copy) and record
what they did in ``afm.uns['mito']``.

The AFM
-------

``X`` holds allele frequencies, and the layers ``AD`` (alternative molecules) and ``DP``
(molecules at the site) hold the counts everything below is computed from. ``var`` has the
position, reference, alternative and base quality of each MT-SNV. For mtscATAC, reads are
converted to molecules from the strand counts (both-strand rule), so that the counts mean the
same thing as in the other two assays. Older AFMs are converted with :func:`mito.io.migrate_afm`.

1. Cells
--------

:func:`mito.pp.filter_cells` keeps the cells with enough MT coverage (``cell_filter``, and
``mean_cov_all``, default 10).

2. Variants: from candidates to characters
------------------------------------------

:func:`mito.pp.filter_afm` runs the stages below in this order. Each one is also a public function.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Stage
     - What it does
     - Functions
   * - Candidates
     - Keeps variants whose per-cell molecule counts are over-dispersed relative to sequencing noise
       (the redeemR Filter-2 test), of enough quality, not known artefacts, at most ``max_candidates``
       (default 1000)
     - ``filter_overdispersed_variants``, ``filter_low_quality_variants``, ``filter_known_artefacts``,
       ``cap_candidates``
   * - Provisional genotypes
     - Stricter calls (``geno_alpha``/10), so the next stage is not run on noise
     - ``call_genotypes``
   * - Clonality
     - Keeps variants whose carriers cluster on the cell kNN graph, or are mutually exclusive with the
       other variants' carriers. Replaces the Moran's I filter of 1.0.0
     - ``filter_non_clonal_variants``
   * - Final genotypes and signal
     - Calls at ``geno_alpha``; signal over background per variant (``.var['snr']``)
     - ``call_genotypes``, ``filter_low_signal_variants``
   * - Imputation (optional)
     - Adds a call to a cell when its neighbourhood, built without that variant, carries it. Off by
       default: it recovers cells on some datasets and costs accuracy on others
     - ``impute_dropouts``
   * - Characters
     - Prevalence cap, at least two calls, four-gamete compatibility; then cell-cell distances
     - ``filter_incompatible_variants``

Genotyping
~~~~~~~~~~

A cell carries a variant if its alternative molecules are too many to come from sequencing error
alone: a one-sided binomial test against the variant's own error rate (``geno_alpha``, default
1e-3). The error rate is estimated on the cells not currently called, and re-estimated until the calls
stop changing, so that a clone does not inflate the background of its own marker. There is no mixture
model and no fixed allele-frequency threshold.

3. Phylogeny
------------

:func:`mito.tl.build_tree` builds a UPGMA tree from the weighted-Jaccard distances between the
genotypes (a Cassiopeia tree). Cells with the same genotype are at distance 0, and average linkage would merge them
one by one in an arbitrary order: such tied merges are collapsed into polytomies (``collapse_ties=True``), and the
branches carry the UPGMA lengths. The cell kNN graph (:func:`mito.pp.kNN_graph`) and the embeddings
(:func:`mito.pp.reduce_dimensions`) use the same distances.

4. Clones
---------

:func:`mito.tl.annotate_clones` turns the tree into discrete clones in a few steps:

1. **Evidence cut** (:func:`mito.tl.evidence_cut`): the tree is split only where the characters pay for
   it. A clade is subdivided only if at least ``min_supported`` of its children carry markers that are
   prevalent inside them and absent from their siblings. A mutation shared by a whole clade is evidence
   for the clade, never for splitting it.
2. **Abstention**: a cell keeps its label only if one of its calls is corroborated by its neighbourhood
   (``tau``). The others are labelled ``unassigned`` instead of being forced into the nearest clade.
3. **Small clones** (optional, ``small_clones=True``): a character carried by 2 to 4 cells that form exactly
   a clade makes them a clone. It gains recall of the smallest clones at some cost in precision, and suits
   assays where single markers are reliable (molecule counts).
4. **Clade support** (``support=True``, default; ``min_support`` optional): the support of every clade, and
   so of every clone, is computed by count resampling (:func:`mito.tl.clade_support`). Clones below
   ``min_support`` are left unassigned.

Labels are written to ``afm.obs['MiTo_clone']``, and the support of each clone is reported with them.

Clade support
~~~~~~~~~~~~~

Trees from a few hundred characters, most private to one clone, make the usual character bootstrap measure
the number of markers of a clade: a clone marked by one variant can never be supported. What is uncertain
is the counts (a cell called on one or two molecules may not be a carrier). Each replicate therefore draws
every cell's alternative molecules again, re-calls the genotypes with the same test, and rebuilds the tree.
A clade's support is its average over the replicates, as a transfer bootstrap (default): the fraction of
its cells that need not move for it to be a clade of the replicate. 100 replicates of a few thousand cells
take seconds to a minute.

Choices that depend on the data
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - When to change it
   * - ``impute=True`` in ``filter_afm``
     - Cells with many dropouts that you would rather assign; check on a dataset with known clones first
   * - ``small_clones=True``
     - Molecule-count assays, if you care about clones of 2-4 cells
   * - ``min_support=0.5``
     - Conservative labelling: leave unassigned the clones that count resampling does not recover
   * - ``max_candidates``
     - Deeply sequenced datasets with many over-dispersed variants (default 1000)
   * - ``geno_alpha``
     - Stricter or more permissive calls (default 1e-3)

For numbers on simulated data, MAESTER, and ReDeeM vs mtscATAC on the same cells, see :doc:`benchmark`.
The full signature and parameters of every function are in the :doc:`api`.

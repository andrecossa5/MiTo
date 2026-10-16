Migrating from 1.0.0
====================

MiTo 2.0.0 is a redesign with a new API. The original API stays available as MiTo 1.0.0, the version used for the
analyses of the MiTo paper (`documentation <https://mito.readthedocs.io/en/v1.0.0/>`_):

.. code-block:: bash

   pip install "scmito<2"    # 1.0.0
   pip install scmito        # 2.x

The two versions give different numbers, because the variant pipeline and the clone annotation changed (see the
CHANGELOG). Pipelines that call the 1.0.0 API, such as nf-MiTo, keep working with ``scmito<2``.

The quick start, side by side
-----------------------------

.. code-block:: python

   # 1.0.0
   afm = mt.pp.filter_cells(afm, cell_filter='filter2')
   afm = mt.pp.filter_afm(afm, filtering='MiTo')
   tree = mt.tl.build_tree(afm, precomputed=True, solver='UPMGA')
   model = mt.tl.MiToTreeAnnotator(tree)
   model.clonal_inference()
   mt.pl.plot_tree(tree, features=['MiTo clone'])

.. code-block:: python

   # 2.x: functions work in place
   mt.pp.filter_cells(afm, cell_filter='filter2')
   mt.pp.filter_afm(afm)
   tree = mt.tl.build_tree(afm, solver='UPMGA')
   mt.tl.annotate_clones(tree, afm)          # labels in afm.obs['MiTo_clone']
   mt.pl.plot_tree(tree, annot='MiTo_clone')

What changed
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 1.0.0
     - 2.x
   * - ``pp.filter_afm(filtering=, filtering_kwargs=, filter_moran=, ...)``
     - ``pp.filter_afm(max_candidates, cand_min_cell_coverage, cand_q_max, cand_min_quality, cand_only_genes,
       filter_artefacts, qc_alpha, geno_alpha, impute, max_prevalence, min_n_var, metric, ncores, seed, copy)``;
       one pipeline for all assays. The stages are also available alone (``pp.filter_overdispersed_variants``,
       ``pp.filter_non_clonal_variants``, ...)
   * - ``pp.call_genotypes(bin_method=, t_vanilla=, min_AD=, t_prob=, min_cell_prevalence=)``
     - ``pp.call_genotypes(afm, alpha, max_iter, copy)``
   * - ``pp.compute_distances(precomputed=, bin_method=, binarization_kwargs=)``
     - ``pp.compute_distances(afm, ...)`` on the genotypes in the AFM
   * - ``pp.kNN_graph(X, D, k=10, from_distances=, nn_kwargs=)``
     - ``pp.kNN_graph(afm, k=15, distance_key, key_added, copy)``: writes ``.obsp`` and ``.uns`` like
       ``sc.pp.neighbors``
   * - ``tl.build_tree(precomputed=, bin_method=, binarization_kwargs=)``
     - ``tl.build_tree(afm, solver=)``
   * - ``tl.MiToTreeAnnotator(tree).clonal_inference()``
     - ``tl.annotate_clones(tree, afm)``; ``small_clones`` and ``min_support`` are options
   * - ``tl.bootstrap_MiTo``, ``tl.bootstrap_bin``
     - ``tl.clade_support``, ``tl.clone_support``
   * - ``pl.plot_tree(features=, ...)``
     - ``pl.plot_tree(annot=, cmaps=, limits=, colorstrips=, branches=, leaves=, internal_nodes=, ...)``; the
       old argument names raise an error that names the replacement
   * - ``io.read_coverage(afm_raw, ...)``
     - ``io.read_coverage(afm, ...)``
   * - ``pp.filter_cell_clones``
     - ``pp.filter_small_clones``
   * - ``io.make_afm``
     - same, with ``multi_allelic_sites`` and ``molecules``; ``io.make_afm_from_counts`` is new

AFMs written by 1.x are converted with :func:`mito.io.migrate_afm`.

Defaults that changed: ``pp.filter_cells`` ``mean_cov_all`` 20 to 10; ``pp.kNN_graph`` ``k`` 10 to 15; ReDeeM
``edge_trim`` 4 to 9.

Removed
-------

* ``pp``: ``filter_MQuad``, ``filter_MiTo``, ``filter_baseline``, ``filter_cell_clones``
* ``tl``: ``MiToTreeAnnotator``, ``agg_pseudobulk``, ``bootstrap_MiTo``, ``bootstrap_bin``, ``nb_regression``
* ``ut``: ``extract_bench_df``, ``fit_betabinom``, ``fit_binom``, ``fit_mixbinom``, ``fit_nbinom``, ``flatten_dict``,
  ``format_rank_genes_groups``, ``format_tuning``, ``genotype_mix``, ``get_posteriors``, ``get_top_markers``, ``ji``,
  ``make_folder``, ``order_groups``, ``perturb_AD_counts``, ``rank_items``, ``run_GSEA``, ``run_ORA``,
  ``select_jobs``, ``subsample_afm``, ``update_params``

New
---

* ``io``: ``make_afm_from_counts``, ``migrate_afm``
* ``pp``: ``cap_candidates``, ``filter_incompatible_variants``, ``filter_known_artefacts``,
  ``filter_low_quality_variants``, ``filter_low_signal_variants``, ``filter_non_clonal_variants``,
  ``filter_overdispersed_variants``, ``filter_small_clones``, ``impute_dropouts``, ``select_gt_enriched_variants``
* ``tl``: ``annotate_clones``, ``clade_support``, ``clone_support``, ``compute_expansions``, ``compute_fitness``,
  ``evidence_cut``, ``rescue_unassigned``, ``small_clone_pass``
* ``ut``: ``dataset_metrics``, ``simulate_afm``

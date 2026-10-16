# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-10-16
One pipeline for the three MT-scLT assays: every AFM holds molecule counts, and
`filter_afm` runs the same stages with the same defaults on MAESTER, ReDeeM and
mtscATAC. 2.0.0 is a redesign with a new API; the original API stays available as 1.0.0
(`pip install "scmito<2"`).

The 0.3.0-0.5.0 development versions were never released: this entry, with "The redesign
(developed as 0.3.0)" below, describes everything that changed since 1.0.0. The results of
the MiTo paper were obtained with the original API (1.0.0; the Nature Communications release
is 0.1.5, see the 1.0.0 entry for the differences): see
[MiTo_benchmark_repro](https://github.com/andrecossa5/MiTo_benchmark_repro).

**Migrating from 1.0.0:** see the migration guide in the documentation.
`mt.pp.filter_baseline` / `filter_afm(filtering='MiTo')` ->
`mt.pp.filter_cells` + `mt.pp.filter_afm`; `mt.tl.MiToTreeAnnotator(tree).clonal_inference()`
-> `mt.tl.annotate_clones(tree, afm)` (labels in `afm.obs['MiTo_clone']`);
`mt.tl.bootstrap_MiTo` / `bootstrap_bin` -> `mt.tl.clade_support`; AFMs written by 1.x ->
`mt.io.migrate_afm`.

### Added
- **Over-dispersion candidate stage** (`mt.pp.filter_overdispersed_variants`), the
  variant-level part of redeemR Filter-2 (MitoDrift's input): >= 2 carriers, no
  homoplasmy, chi-square test of the per-cell counts against one binomial at the
  site's depth (Storey q <= .05), blacklisted regions and RSRS50 ancestral variants
  removed, median depth >= 5. Markers of two-cell clones stay testable.
- **`max_candidates=1000`** in `filter_afm` (`mt.pp.cap_candidates`): at most this
  many candidates, the most over-dispersed, with a warning.
- **mtscATAC as molecule counts.** `make_afm` converts mgatk read counts for
  mtscATAC (`molecules=None` -> on), from the strand counts alone: the reads per
  molecule f and the fraction of molecules read by both mates o = f - 1 are
  estimated from the strand variance (1.66 on pL1000, 1.73 from ReDeeM molecules of
  the same cells); coverage = reads / f; alternative molecules = min(forward,
  reverse) when o >= 0.5, rint(reads / f) otherwise (single-end reads, long
  fragments). Stored in `.uns["molecules"]`; libraries with PCR duplicates left in
  are detected and reported.
- **Clade support from count resampling** (`mt.tl.clade_support`): each replicate
  draws every cell's alternative molecules again, re-calls genotypes and rebuilds the
  UPGMA tree; support is the transfer bootstrap expectation (Lemoine et al., 2018;
  `method="exact"` for Felsenstein's). 100 replicates: 0.3 s (340 cells) to 13 s
  (2,400 cells). Written on the tree nodes. `annotate_clones` computes it when the
  tree has none (`support=True`, `n_boot=100`) and reports each clone's support;
  `annotate_clones(min_support=...)` leaves weakly supported clones unassigned (off by
  default). On pL1000 ReDeeM, clade precision of the tree .49 -> .81 at support
  >= .5 (recall .65 -> .59).
- **Small-clone pass** (`mt.tl.small_clone_pass`, `annotate_clones(small_clones=True)`,
  off by default): a character carried by 2-4 cells that are exactly a clade becomes
  a clone.
- `make_afm(multi_allelic_sites=True)`: keep every alternative allele of a site.
- **`mt.io.make_afm_from_counts`**: an AFM from long tables of alternative counts and
  coverage per cell and site, for data from other pipelines; same contract and rules as
  the readers. The RedeemV reader also accepts gzipped files.
- **Documentation**: Getting started (simulated flat and nested clones), Loading data
  (the AFM; maegatk, mgatk, RedeemV and counts tables, on real 50-cell subsets shipped
  in `docs/source/data`) and Benchmark (simulated, MAESTER, ReDeeM vs mtscATAC on the
  same cells) notebooks.

### Changed
- `tl.build_tree(solver="UPMGA")` collapses tied merges into polytomies (`collapse_ties=True`; `False` keeps the
  binary tree) and sets UPGMA branch lengths. Cells with the same genotype were merged one by one in an arbitrary order,
  which drew long ladders; the clades and the clones are unchanged.
- `pl.plot_tree` draws the true branch lengths by default for ultrametric trees with varying lengths (UPGMA), with colorstrips
  scaled to the tree (`extend_branches=None`; `True` restores equal-length branches), and accepts zero-length branches.
- `pl.plot_tree`: `leaf_length` (fans for polytomies), `branches={"annot": ...}` colors the leaf branches by clone, and
  `collapse_clones="MiTo_clone"` draws each single-label clade as a short fan (`clone_style="fan"`; `"wedge"` for a
  filled triangle, `"full"` for a fan from the MRCA), which keeps large trees readable. `gap_by="MiTo_clone"` (with `gap`, in cell widths) opens space between neighbouring clones.
- New `pl.clone_packing`: clones as circles (area = cells) packed inside the clades of the tree that hold them, colored by a
  covariate. Clone-level view of a phylogeny with hundreds of clones.
- New `pl.plot_clone_tree`: one tip per clone, a circle whose area is its cells, on the tree with its true branch lengths;
  colored by clone or by a per-clone value.
- **Clonality filter 10-26x faster, in less memory** (and imputation with it): the
  leave-one-out kNN graphs are updated incrementally (only the held-out variant's
  carriers change; exact), and the null sets come from one bank of stratified
  permutations (p-values independent of variant order). 1,000 candidates x 3,300
  cells: 87 s -> 3.3 s; 9,700 cells: 70 s / 4.6 GB -> 6.8 s / 1.3 GB.
- **maegatk / mgatk reader: same AFM, ~3x faster, ~6x less memory.** Tables are read
  typed, column-selective and in chunks, reduced to alternative basecalls as they are
  read, and assembled straight into sparse matrices. MDA_PT (4,761 cells): 64 s / 10 GB
  -> 18 s / 2.7 GB; pL1000 mgatk: 71 s / 10 GB -> 21 s / 1.7 GB. Checked identical slot by
  slot (benchmarks/reader_check.py).
- **`build_tree(solver="UPMGA")` uses scipy's average linkage** (cassiopeia for the
  other solvers): 32.5 s -> 0.18 s for 2,400 cells.
- The clones table of `annotate_clones` gives each clone's clade as the common
  ancestor of its cells (the cut could stop at a sub-clade of a clone), and its support.
- **Reader**: one alternative allele per site by default, the dominant one;
  `migrate_afm` applies the same rule to AFMs written by MiTo < 0.3
  (`multi_allelic_sites=True` keeps every allele).
- RedeemV reader: `edge_trim=9` by default (redeemR; was 4).
- `filter_cells(mean_cov_all=10)` (was 20): redeemR's qualified cells. MAESTER's
  `filter2` is unaffected.
- `filter_afm(filter_artefacts=None)`: dbSNP / REDIdb filter on for MAESTER (RNA)
  only.
- The signal-over-background ratio is annotated (`.var["snr"]`, used to rank
  conflicting characters) but no longer filtered on.

### Fixed
- Clonality filter and imputation: the held-out variant leaked into its own graph.
  The Gram matrix was computed in float32 and down-dated in float64; for cells whose
  only variant was the held-out one, the residue (~1e-6) made them each other's
  nearest neighbours, so sole-marker variants could confirm themselves.
- `weighted_jaccard`: no 0/0 warning for cells without calls.

### Removed
- `mt.tl.bootstrap_MiTo` and `mt.tl.bootstrap_bin` (character resampling): replaced
  by `mt.tl.clade_support`.
- The read-threshold candidate stage: `mt.pp.filter_candidate_variants` and the
  `filter_afm` parameters `cand_min_site_cov`, `cand_min_frac_negative`,
  `cand_min_n_positive`, `cand_af_confident`, `cand_min_n_confident`,
  `cand_min_AD_in_positives`, `cand_min_DP_in_positives`, and `min_snr`.
- The multi-allelic site drop of `filter_low_quality_variants` and of the AFM
  builder (the reader decides, `multi_allelic_sites`).
- Unused helpers `mt.ut.ji`, `mt.ut.flatten_dict`, `mt.ut.update_params`.

### The redesign (developed as 0.3.0, 2026-09-21)
#### Changed
- **MT-only scope.** MiTo now covers MAESTER, mtscATAC and ReDeeM data
  (`pp_method`: `maegatk`, `mgatk`, `redeem-v`); the Cas9, scWGS and EPI-clone
  readers and their metrics are gone. Functions dispatch on what an AFM carries
  rather than on `uns['scLT_system']`, so a new pre-processing pipeline only has
  to produce the object contract.
- **AFM contract.** One dense integer `DP` layer holds the coverage of each
  variant's site, defined for every cell; `site_coverage` and the `qual` layer
  are gone (base quality is a `.var` column). An unfiltered MAESTER AFM shrank
  from 123 MB to ~24 MB.
- **`filter_afm` implements the new variant pipeline**: candidate MT-SNVs ->
  read-level genotyping against each variant's own error rate -> clonality QC on
  the cell kNN graph -> signal over background -> optional dropout imputation ->
  prevalence / four-gamete characters -> distances. 16 documented parameters,
  keyword-only, in pipeline order.
- **`mt.tl.annotate_clones`** replaces `MiToTreeAnnotator.clonal_inference`: the
  tree is cut where the characters pay for the split, and cells whose calls their
  neighbourhood does not corroborate are left `unassigned`.
- **`mt.pl.plot_tree` rewritten**: annotations are named once in `annot`
  (metadata column or character), coloured through `cmaps` / `limits`, and each
  tree element is configured by one dictionary. 44 parameters -> 14, with
  renamed arguments reported by name.
- **scverse conventions**: `mt.pp` / `mt.tl` functions modify the AnnData in
  place and take `copy=`; `mt.pp.kNN_graph` writes `.obsp` / `.uns` like
  `sc.pp.neighbors`; provenance lives in one `.uns['mito']` namespace, one
  record per public function.

#### Added
- `mt.pp.call_genotypes` (per-cell binomial test against the variant's own,
  iteratively estimated error rate), `filter_non_clonal_variants` (join count /
  exclusivity on the leave-one-out kNN graph), `filter_low_signal_variants`,
  `impute_dropouts`, `filter_incompatible_variants`.
- `mt.tl.evidence_cut`, `clone_support`, `rescue_unassigned`, `compute_fitness`,
  `compute_expansions`; `mt.ut.dataset_metrics`.
- `mt.io.migrate_afm`, which converts an AFM written by MiTo < 0.3 to the current
  contract. Genotyping refuses to run on an unconverted object rather than use a
  masked coverage layer as its denominator.

#### Removed
- Feature-selection strategies superseded by the clonality QC: `MQuad`,
  `weng2024`, `miller2022`, `CV`, and the Moran's I filter (it included the
  variant under test in the distances, so 96% of scattered noise passed).
- The binomial-mixture genotyper (it called cells with zero alternative reads),
  `mt.tl._classification`, `mt.ut.de_utils`, `mt.ut.stats_utils` and unused
  helpers; dependencies `mquad`, `bbmix`, `lightgbm`, `shap`, `gseapy`.

#### Maintenance
- CI actions and pre-commit hooks updated (`download-artifact`,
  `action-gh-release`, `pyproject-fmt`, `ruff-pre-commit`).
- Tutorials re-executed against the new API; the suite is 347 tests and needs no
  external data.

#### Fixed
- `multipletests` is imported from `statsmodels.stats.multitest`; the
  `statsmodels.sandbox` path it used was removed in statsmodels 0.15, which had
  been breaking every CI run since 31 Aug 2026 on unchanged code.
- One character without a positive allele frequency made *every* cell-cell
  distance NaN; weights are now taken over called cells, with a fallback.
- The four-gamete filter resolved conflicts by count alone, deleting prevalent
  real markers in favour of sparse noise (613 of 1272 genotyped cells lost their
  only character on one dataset); it now drops the variant with the least signal.
- Bootstrap replicates dropped the genotypes, so `compute_distances` failed on
  them; they are now seeded and carry every layer.
- Plotting: categorical legends in `draw_embedding`, every `layer=` in
  `heatmap_variants`, `matplotlib.cm.get_cmap` (removed in matplotlib 3.9),
  axes creation when `ax` is not given.

## [1.0.0] - 2026-10-16
First stable release of the original MiTo API (`mt.pp.filter_baseline`, `mt.pp.filter_afm(filtering=...)`,
`mt.tl.MiToTreeAnnotator`, ...), the one used for the analyses of the MiTo paper. The code is that of 0.2.1; 1.0.0
adds the matched documentation. 0.2.1 stays published (same code).

MiTo 2.x has a redesigned API. The 0.3.0-0.5.0 development versions were never released: the redesign ships as 2.0.0.

### Changed
- Version 1.0.0, classifier "Production/Stable".
- Documentation links point to `mito.readthedocs.io` (the previous `scmito.readthedocs.io` does not exist), in the
  README, the package metadata and the tutorial.
- The tutorial is re-executed with 1.0.0 on current dependencies (numpy 2, pandas 3, scipy 1.18): its results are
  unchanged. The Moran's I variant filter is a permutation test without a fixed seed, so cell and variant counts can
  differ by a unit between runs.
- Docstrings match their signatures: `filter_cells` (`mean_cov_all`, default `cell_filter`), `reduce_dimensions`
  (`ncores`, return value), `plot_tree` (`vmin`, `vmax`, `x_space`) and `nb_regression` (`df`, and it returns a
  DataFrame).
- README: versions note, "Reproducing the paper" section.
- Dependency ceilings: numpy<3, scipy<2, pandas<4, scikit-learn<2, matplotlib<4, scanpy<2 and statsmodels<0.16, so a
  breaking release of the core stack cannot break new installs of this version. Raised once a newer release is verified.

### Differences from 0.1.5 (the Nature Communications release), as of 0.2.0
- `mt.pp.kNN_graph(from_distances=True)` returns the `k` nearest neighbours of each cell; 0.1.5 returned every other
  cell. kNN-based quantities (kNN purity, entropy and kBET, UMAP embeddings, Leiden clustering on distances) differ
  from the published ones.
- `annotate_vars`: `median_af_in_positives` (a mean over the positive cells, whatever its name) is computed on their
  allele frequencies; 0.1.5 computed it on a boolean matrix, so it was constant.
- The `genotype_MiTo_smooth` function and the `"MiTo_smooth"` genotyping option are removed.

## [0.2.1] - 2026-10-15
### Fixed
- Importable with statsmodels >= 0.15: `multipletests` is imported from
  `statsmodels.stats.multitest` (`statsmodels.sandbox.stats.multicomp` no longer
  provides it), in `mt.pp`, `mt.tl.annotate` and `mt.tl.phenotype`. No change to any
  method: results are those of 0.2.0.
- `mt.pl.plot_tree`: no longer calls `matplotlib.cm.get_cmap` (removed in matplotlib 3.9)
  on a colormap it had already looked up.

## [0.2.0] - 2026-08-05
### Added
- Distributed on PyPI as `scmito` (previously `mito-utils`); import name is
  unchanged (`import mito as mt`).
- Test-suite of 462 tests, and CI across Linux and macOS on Python 3.11-3.13.
- Automated PyPI releases and GitHub Releases via Trusted Publishing.

### Changed
- Modern `pyproject.toml` packaging (hatchling); `setup.py` removed.
- Requires Python 3.11+; phylogenetics now depends on `cassiopeia-mt`.
- Assets ship inside the package and are resolved with `importlib.resources`;
  installed footprint down from 14 MB to ~1.5 MB.
- Tree solvers reduced to `UPMGA`, `NJ`, `spectral`, `greedy`.
- Genotyping method `MiTo_smooth` removed.

### Fixed
- `median_af_in_positives` averaged a boolean mask, so it was always 1.0.
- Every distance metric except `weighted_jaccard` raised on current
  scikit-learn and SciPy.
- `k` was ignored when building a kNN graph from precomputed distances.
- `kbet`, `NN_entropy` and `NN_purity` raised on a pandas Series of labels.
- `reduce_dimensions(method='UMAP')` crashed on disconnected graphs.
- `plot_tree` crashed with `add_root=True` and with `ax=None`.
- `draw_embedding` crashed when given a named palette.
- `fit_mixtures` computed `deltaBIC` and then discarded it.
- Filters that select nothing now raise a clear error instead of failing deep
  inside NumPy or SciPy.

## [0.1.5] - 2026-02-03
- Nat Comm release

## [0.1.4] - 2026-01-01
- Last bug fixes

## [0.1.3] - 2026-01-29
### Enhanced
- **kNN Graph Computations**: Improved k-nearest neighbor graph algorithms
  - Optimized distance calculations for better performance in large datasets
  - Enhanced graph construction methods for more accurate neighborhood detection
  - Improved memory efficiency in graph building operations
  - Better handling of edge cases in sparse data scenarios

### Technical
- Refined algorithms for more robust graph-based analyses
- Enhanced computational efficiency for scalable graph operations

## [0.1.2] - 2025-10-23
### Enhanced
- **Performance Optimizations**: Major parallelization improvements using joblib
  - Parallelized Moran's I computation in `filter_variant_moransI` with batch processing
  - Parallelized mutation enrichment computation in `MiToTreeAnnotator.get_M` method
  - Memory-efficient matrix caching to avoid redundant serialization across workers
  - Configurable core usage and temporary folder management for large datasets

### Added
- **Multi-allelic Site Filtering**: New quality control functionality
  - Added `filter_multiallelic_sites` function to remove variants from sites with multiple alleles
  - Integrated multi-allelic filtering in RedeeM data processing pipeline
  - Ensures each genomic position has only one variant type for cleaner analyses
- **Dual Implementation Support**: Both serial and parallel versions available for performance testing
- **Enhanced RedeeM Support**: Improved data processing for RedeeM scLT system

### Fixed
- Memory usage optimization in parallel processing by caching matrix variables
- Proper joblib backend configuration for stable parallel execution
- Improved progress tracking for long-running computations

### Technical
- Migrated from multiprocessing to joblib for better memory management
- Added batch processing patterns for scalable parallel computation
- Enhanced error handling and progress reporting in parallel workflows

## [0.1.1] - 2025-10-08
### Enhanced
- Improved clonal inference algorithm with edge case handling for small phylogenies
- Added `af_treshold` parameter to `resolve_ambiguous_clones` for better clone merging control
- Split `infer_clones` and `resolve_ambiguous_clones` methods for better modularity
- Enhanced grid search optimization with better parameter handling
- Fixed root node exclusion in `_find_clones` to prevent single-clone edge cases

### Fixed
- Resolved issue where very small phylogenies would only return root as single clone
- Improved silhouette score calculation for edge cases with minimal clones
- Better parameter validation and error handling in clonal inference pipeline

### Refactored
- Cleaner separation between clone detection and clone resolution phases
- Improved code organization in `MiToTreeAnnotator` class
- Better debugging support with modular function structure

## [0.1.0] - 2025-10-06
### Added
- Complete documentation overhaul with nbsphinx integration
- Interactive Jupyter notebook tutorial with full cell outputs and visualizations
- Comprehensive getting started guide with MiTo workflow examples
- Hierarchical documentation structure for better navigation
- ReadTheDocs integration with automated builds

### Improved
- Streamlined installation guide with clear step-by-step instructions
- Enhanced plotting library examples and demonstrations
- Better code organization and documentation structure
- Cleaner repository structure with proper .gitignore rules

### Fixed
- nbsphinx configuration for proper notebook rendering
- Image generation and display in documentation
- Documentation build process for ReadTheDocs compatibility
- File size issues with test data exclusion

## [0.0.8] - 2025-09-25
### Fixed
- Add mt.io.make_afm behavior

## [0.0.7] - 2025-09-25
### Fixed
- Add chrM.fa to assets

## [0.0.6] - 2025-09-17
### Fixed
- Fixed asset path detection for conda environments
- Assets now properly accessible via sys.prefix location
- Improved _find_assets_path() function to check conda environment directory

## [0.0.4] - 2025-09-12
### Fixed
- Fixed asset files inclusion in package distribution (PyPI release)
- Improved asset path detection for both development and installed environments
- Assets (dbSNP_MT.txt, REDIdb_MT.txt, formatted_table_wobble.csv, weng2024_mut_spectrum_ref.csv) now properly included in pip installations

## [0.0.3] - 2025-09-11
### Added
- Code refactoring and improvements
- Enhanced functionality and bug fixes
- Updated documentation

### Fixed
- Assets (dbSNP_MT.txt, REDIdb_MT.txt, formatted_table_wobble.csv, weng2024_mut_spectrum_ref.csv) now properly included in pip installations
- Smart asset path finder that works in both development and production environments

## [0.0.2] - 2025-03-25
### Added
- Updated docs.

## [0.0.1] - 2025-03-24
### Added
- Initial release of the mito package.
- Packaging via `setup.py` for PyPI distribution.
- Core functionality for mito analyses.
- First docs.


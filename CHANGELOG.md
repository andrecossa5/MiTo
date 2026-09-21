# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-09-21
### Changed
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

### Added
- `mt.pp.call_genotypes` (per-cell binomial test against the variant's own,
  iteratively estimated error rate), `filter_non_clonal_variants` (join count /
  exclusivity on the leave-one-out kNN graph), `filter_low_signal_variants`,
  `impute_dropouts`, `filter_incompatible_variants`.
- `mt.tl.evidence_cut`, `clone_support`, `rescue_unassigned`, `compute_fitness`,
  `compute_expansions`; `mt.ut.dataset_metrics`.
- `mt.io.migrate_afm`, which converts an AFM written by MiTo < 0.3 to the current
  contract. Genotyping refuses to run on an unconverted object rather than use a
  masked coverage layer as its denominator.

### Removed
- Feature-selection strategies superseded by the clonality QC: `MQuad`,
  `weng2024`, `miller2022`, `CV`, and the Moran's I filter (it included the
  variant under test in the distances, so 96% of scattered noise passed).
- The binomial-mixture genotyper (it called cells with zero alternative reads),
  `mt.tl._classification`, `mt.ut.de_utils`, `mt.ut.stats_utils` and unused
  helpers; dependencies `mquad`, `bbmix`, `lightgbm`, `shap`, `gseapy`.

### Maintenance
- CI actions and pre-commit hooks updated (`download-artifact`,
  `action-gh-release`, `pyproject-fmt`, `ruff-pre-commit`).
- Tutorials re-executed against the new API; the suite is 347 tests and needs no
  external data.

### Fixed
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

## [0.2.1] - 2026-08-06
### Fixed
- Documentation URL in the package metadata pointed at a Read the Docs slug
  that does not exist yet, so the "Documentation" link on PyPI was dead.

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


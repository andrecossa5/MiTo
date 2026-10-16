# Changelog

All notable changes to this project will be documented in this file.

## [1.0.2] - 2026-10-16
Last release of the original MiTo API (the "legacy" line). It supports the analyses of the MiTo paper
([MiTo_benchmark_repro](https://github.com/andrecossa5/MiTo_benchmark_repro)) and the
[nf-MiTo](https://github.com/andrecossa5/nf-MiTo) pipeline. 1.0.1 was not published.

### Fixed
- `mt.ut.extract_kwargs` no longer returns `k`, `gamma` and `resample`, the options of the `MiTo_smooth` genotyping
  method removed in 0.2.0. `filter_afm(**extract_kwargs(args))`, the call the nf-MiTo scripts make, failed with
  `call_genotypes() got an unexpected keyword argument 'k'`. Command lines and tuning tables that still carry
  those options are accepted, and the values ignored.

### Changed
- Documentation and README for 1.0.2: versions and links; the tutorial is re-executed.

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
- `mt.pp.kNN_graph(from_distances=True)` returns the `k` nearest neighbours of each cell; 0.1.5 had a bug returning every other
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


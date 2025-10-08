# Changelog

All notable changes to this project will be documented in this file.

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


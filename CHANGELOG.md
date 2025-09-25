# Changelog

All notable changes to this project will be documented in this file.

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


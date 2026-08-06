<div align="center">
  <img src="image/MiTo_logo_transparent.png" alt="MiTo" width="260">
</div>

# MiTo (Mitochondrial single-cell lineage tracing Toolkit)

**Mitochondrial single-cell multi-omics in Python.**

[![PyPI](https://img.shields.io/pypi/v/scmito.svg)](https://pypi.org/project/scmito/)
[![Python](https://img.shields.io/pypi/pyversions/scmito.svg)](https://pypi.org/project/scmito/)
[![Tests](https://github.com/andrecossa5/MiTo/actions/workflows/test.yml/badge.svg)](https://github.com/andrecossa5/MiTo/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/andrecossa5/badge/?version=latest)](https://andrecossa5.readthedocs.io/en/latest/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MiTo** infer clonal ancestries in single-cell data from natural mtDNA variation. 

The framework provides the infrastructure to convert MAESTER, scmtATAC-seq and RedeeM pre-processing outputs
into Allele Frequency Matrices (i.e., [AnnData](https://anndata.readthedocs.io/) objects), filter informative mtDNA variants, genotype individual cells, and
infer mitochondrial phylogenies and clones for dowstream multi-omic analysis.

## Installation

```bash
pip install scmito
```

Requires Python 3.11+.

```python
import mito as mt
print(mt.__version__)
```

## Quick start

```python
import scanpy as sc
import mito as mt

# Allele Frequency Matrix (cell x site AnnData with AD / DP layers)
afm = sc.read('afm_unfiltered.h5ad')

# Cell and variant filters, distances in mtDNA mutation space
afm = mt.pp.filter_cells(afm, cell_filter='filter2')
afm = mt.pp.filter_afm(afm, filtering='MiTo')

# Phylogeny and clonal inference
tree = mt.tl.build_tree(afm, precomputed=True, solver='UPMGA')
annotator = mt.tl.MiToTreeAnnotator(tree)
annotator.clonal_inference()

# Visualization
mt.pl.plot_tree(tree, features=['MiTo clone'])
```

See [getting-started tutorial](https://andrecossa5.readthedocs.io/en/latest/getting_started_tutorial.html) for 
the full vignette.

## API

MiTo follows the `scverse` layout, composing with `scanpy` and `anndata`:

| Module | Purpose |
| --- | --- |
| `io` | Build AFMs from pre-processing outputs, read/write Newick tree objects |
| `pp` | Cell and variant filtering, genotying, distances, kNN graphs, embeddings |
| `tl` | Tree building, clonal inference, clustering, fate bias |
| `pl` | Trees, heatmaps, embeddings, coverage and variant-spectrum plots |
| `ut` | Metrics, helpers |

**Supported platforms:** scRNA-seq (MAESTER), scmtATAC-seq and 10x MultiOme (RedeeM).

Full reference: [MiTo docs](https://andrecossa5.readthedocs.io/en/latest/index.html).

## Development

```bash
git clone https://github.com/andrecossa5/MiTo.git
cd MiTo
pip install -e ".[test]"
pytest
```

## Citation

If MiTo is useful in your work, please cite:

> Cossa, A. Dalmasso A. et al. *MiTo: mitochondrial lineage tracing and single-cell multi-omics.*
> Nat Comm (2026). [https://doi.org/10.1038/s41467-026-71607-5](https://www.nature.com/articles/s41467-026-71607-5)

MiTo builds on the [Cassiopeia](https://github.com/YosefLab/Cassiopeia) package for phylogeny reconstruction and data infrastructure. Please, cite it as well.

## Releases

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
  <img src="image/MiTo_logo_transparent.png" alt="MiTo" width="260">
</div>

# MiTo (Mitochondrial single-cell lineage tracing Toolkit)

**Mitochondrial single-cell multi-omics in Python.**

[![PyPI](https://img.shields.io/pypi/v/scmito.svg)](https://pypi.org/project/scmito/)
[![Python](https://img.shields.io/pypi/pyversions/scmito.svg)](https://pypi.org/project/scmito/)
[![Tests](https://github.com/andrecossa5/MiTo/actions/workflows/test.yml/badge.svg)](https://github.com/andrecossa5/MiTo/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/mito/badge/?version=latest)](https://mito.readthedocs.io/en/latest/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MiTo** infers clonal ancestries in single-cell data from natural mtDNA variation.

The framework converts MAESTER, scmtATAC-seq and ReDeeM pre-processing output into Allele
Frequency Matrices ([AnnData](https://anndata.readthedocs.io/) objects), selects the mtDNA
variants that mark lineages, genotypes individual cells, and infers mitochondrial
phylogenies and clones for downstream multi-omic analysis.

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
import mito as mt

# Allele Frequency Matrix from pre-processing output (molecule counts, AD / DP layers)
afm = mt.io.make_afm('path/to/maegatk_output', scLT_system='MAESTER', pp_method='maegatk')

# Cell and variant filters, single-cell genotypes, distances in mtDNA mutation space
mt.pp.filter_cells(afm, cell_filter='filter2')     # 'filter1' (default) for ReDeeM / mtscATAC
mt.pp.filter_afm(afm)

# Phylogeny, clade support (count resampling) and clones
tree = mt.tl.build_tree(afm, solver='UPMGA')
mt.tl.clade_support(tree, afm)                    # optional: support of every clade
mt.tl.annotate_clones(tree, afm)                  # min_support=0.5 for higher precision

# Visualization
mt.pl.plot_tree(tree, annot='MiTo_clone')
```

The same calls, with the same defaults, run on the three assays:

| Assay | `make_afm(scLT_system=, pp_method=)` | What the reader does |
| --- | --- | --- |
| MAESTER | `'MAESTER'`, `'maegatk'` | UMI consensus counts, as they are |
| ReDeeM | `'ReDeeM'`, `'redeem-v'` | consensus molecules, 9 bp trimmed from fragment ends |
| mtscATAC | `'mtscATAC'`, `'mgatk'` | reads to molecules: alternative molecules read by both mates, coverage / reads per molecule (estimated from the strand counts) |

`filter_afm` then selects variants whose per-cell counts are over-dispersed relative to
sequencing noise (redeemR Filter-2, at most `max_candidates=1000`), keeps those whose
carriers are clonal on the cell kNN graph, and genotypes each cell against each
variant's own error rate.

See the [getting started tutorial](https://mito.readthedocs.io/en/latest/getting_started.html)
for the full vignette, and the [ground truth benchmark](https://mito.readthedocs.io/en/latest/benchmark.html)
for a worked example on real MAESTER data.

## API

MiTo follows the `scverse` layout, composing with `scanpy` and `anndata`:

| Module | Purpose |
| --- | --- |
| `io` | Build AFMs from pre-processing output, read/write Newick trees |
| `pp` | Cell and variant filtering, genotyping, distances, kNN graphs, embeddings |
| `tl` | Tree building, clade support, clonal annotation, fate bias |
| `pl` | Trees, heatmaps, embeddings, coverage and variant-spectrum plots |
| `ut` | Metrics, MT annotations, simulation, helpers |

Pre-processing is done by [nf-MiTo](https://github.com/andrecossa5/nf-MiTo).

Full reference: [MiTo docs](https://mito.readthedocs.io/en/latest/index.html).

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

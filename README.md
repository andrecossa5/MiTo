# MiTo
This is `MiTo`: a python package for robust inference of mitochondrial clones and phylogenies.
See also [nf-MiTo](https://github.com/andrecossa5/nf-MiTo), the companion Nextflow pipeline.

## Documentation
A preliminary documentation of key functionalitites and APIs is available at [MiTo Docs](https://andrecossa5.readthedocs.io/en/latest/index.html).

## Installation
1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) (or conda)
2. Clone this repo:

```bash
git clone https://github.com/andrecossa5/MiTo.git
```

3. Reproduce MiTo conda environment:

```bash
cd MiTo
mamba env create -f ./envs/environment.yml -n MiTo
```

3. Activate the environment, and install MiTo via pypi:

```bash
mamba activate MiTo
pip install .
```

4. Verify successfull installation:

```python
import mito as mt
```

## Releases
See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.

# MiTo
This is the first release of `MiTo`: a python package for robust inference of mitochondrial clones and phylogenies.
See `nf-MiTo`, the companion Nextflow pipeline, at [https://github.com/andrecossa5/nf-MiTo].

## Documentation
A comprehensive documentation of key functionalitites and APIs is in the making.

## Installation
1. Install mamba (or conda) (https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Reproduce the environment with all needed dependencies at once, with:

```bash
mamba env create -f ./envs/environment.yml -n <your conda env name>
```

3. Activate the environment and manually install cassiopeia-lineage (no dependencies: it is all already present in the environment):

```bash
mamba activate <your conda env name>
pip install --no-deps git+https://github.com/YosefLab/Cassiopeia.git@e7606afd10035a75f718ffb988666264e721700e
```

4. Install `MiTo`:

```bash
pip install mito-utils
```

5. Verify successfull installation:

```python
import mito as mt
```

## Releases
See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.


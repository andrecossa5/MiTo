# MiTo
This is the first release of `MiTo`: robust inference of mitochondrial clones and phylogenies.

## Documentation
See ... for a comprehensive description of key functionalitites, main APIs, and tutorials.

## Installation
1. Install mamba (or conda) package manager (https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Reproduce the necessary computational environment with:

```bash
mamba env create -f ./envs/environment.yml -n <your conda env name>
```

3. Activate the environment, and manually install cassiopeia-lineage (no dependencies: all has been 
already installed in the environment. Sometimes pip messes with cassipeia dependencies trying to re-install them).

```bash
mamba activate <your conda env name>
pip install --no-deps git+https://github.com/YosefLab/Cassiopeia.git@e7606afd10035a75f718ffb988666264e721700e
```

4. Install `MiTo`:

```bash
pip install mito==0.0.1
```

5. Verify successfull installation:

```python
import mito as mt
```

## Releases
See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.


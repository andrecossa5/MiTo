<div align="center">

<img src="image/nf_mito.png" alt="MiTo Logo" width="500">

# 🔴 **MiTo**

*Mitochondrial lineage tracing and single-cell multi-omics in Python*

---

[![Nextflow](https://img.shields.io/badge/nextflow-%E2%89%A522.04.0-brightgreen.svg)](https://www.nextflow.io/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.06.17.660165-blue)](https://doi.org/10.1101/2025.06.17.660165)

</div>

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
mamba env create -f envs/environment.yml -n MiTo
```

3. Activate the environment, and install MiTo via pypi:

```bash
mamba activate MiTo
pip install mito_utils
```

4. Verify successfull installation:

```python
import mito as mt
```

## Releases
See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.

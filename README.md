<div align="center">

<img src="image/nf_mito.png" alt="nf-MiTo Logo" width="500" style="margin-bottom: 30px;">

<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a85b6 100%); padding: 60px 40px; border-radius: 25px; margin: 30px auto; max-width: 1000px; box-shadow: 0 20px 50px rgba(0,0,0,0.3); border: 4px solid rgba(255,255,255,0.15);">
  
  <h1 style="border: none; font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', 'Source Code Pro', 'Menlo', 'Consolas', monospace; font-size: 6em; font-weight: 900; margin: 20px 0; letter-spacing: 0.05em; text-transform: none; line-height: 1;">
    <span style="color: #dc2626; filter: drop-shadow(3px 3px 6px rgba(0,0,0,0.4));">MiTo</span>
  </h1>
  
  <div style="width: 200px; height: 6px; background: linear-gradient(90deg, #ff6b6b, #ff8e53, #4ecdc4, #44a08d); margin: 35px auto; border-radius: 3px; box-shadow: 0 3px 12px rgba(0,0,0,0.4);"></div>
  
  <p style="font-size: 1.5em; font-weight: 400; color: #f8f9fa; margin: 30px 0 10px 0; line-height: 1.6; font-style: italic; text-shadow: 2px 2px 4px rgba(0,0,0,0.4); max-width: 800px; margin-left: auto; margin-right: auto;">
    Mitochondrial lineage tracing and single-cell multi-omics in Python
  </p>
  
</div>

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/conda-enabled-green.svg)](https://conda.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.06.17.660165-blue)](https://doi.org/10.1101/2025.06.17.660165)

</div>

## Documentation
An extensive documentation of MiTo's key functionalitites is available at [MiTo Docs](https://andrecossa5.readthedocs.io/en/latest/index.html).

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

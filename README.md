# MiTo

The MiTo package is still under active development, and is proximal to the first release. To use it, perform the following:

1. Install mamba package manager (https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Choose one of the following environment recipes:

```bash
envs
├── environment.yml # minimal, OSX
├── environment_latest_OSX.yml # Latest, OSX
├── environment_mito_utils.yml # Verified, Linux
└── environment_preprocessing.yml # Minimal, only for preprocessing
```

3. Reproduce the environment with:

```bash
mamba env create -f <chosen environment name>.yml -n <your conda env name>
```

4. cassiopeia and MQuad packages need manual installation. After reproducing the environment:

```bash
mamba activate <chosen environment name>
pip install mquad
pip install git+https://github.com/YosefLab/Cassiopeia@master#egg=cassiopeia-lineage
```

5. After these steps, activate the new environment and cd to the main mito_utils repo path.
Make the repo path visible to python import system.

```path
cd <path to mito_utils clone>
mamba develop .
```

6. Fire a python terminal and verify successfull installation of all packages.

```python
import mito as mt
```


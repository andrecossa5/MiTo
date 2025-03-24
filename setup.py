# Setup.py
from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mito-utils",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Utilities for MT-based single cell Lineage Tracing (scLT).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrecossa5/MiTo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        # Note: for best reproducibility the environment must
        # be setup before hand, as indicated in the README.md
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "sphinxcontrib-napoleon",
        ],
    },
    include_package_data=True,
)

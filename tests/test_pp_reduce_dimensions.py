"""
mito.pp.reduce_dimensions

Light coverage: the heavy input validation happens upstream in filter_afm, so
this checks the embedding methods, their output slots and the main knobs.
"""

import numpy as np
import pytest
import mito as mt


METHODS = ["PCA", "UMAP", "diffmap"]
SLOTS = {"PCA": "X_pca", "UMAP": "X_umap", "diffmap": "X_diffmap"}


@pytest.mark.parametrize("method", METHODS)
def test_each_method_writes_its_obsm_slot(afm_filtered, method):
    mt.pp.reduce_dimensions(afm_filtered, method=method, ncores=1)
    assert SLOTS[method] in afm_filtered.obsm


@pytest.mark.parametrize("method", METHODS)
def test_embedding_shape(afm_filtered, method):
    mt.pp.reduce_dimensions(afm_filtered, method=method, n_comps=2, ncores=1)
    X = afm_filtered.obsm[SLOTS[method]]
    assert X.shape[0] == afm_filtered.shape[0]
    assert X.shape[1] >= 2


@pytest.mark.parametrize("n_comps", [2, 3, 5])
def test_n_comps(afm_filtered, n_comps):
    mt.pp.reduce_dimensions(afm_filtered, method="PCA", n_comps=n_comps, ncores=1)
    assert afm_filtered.obsm["X_pca"].shape[1] >= n_comps


@pytest.mark.parametrize("method", METHODS)
def test_embedding_is_finite(afm_filtered, method):
    mt.pp.reduce_dimensions(afm_filtered, method=method, ncores=1)
    assert np.isfinite(afm_filtered.obsm[SLOTS[method]]).all()


@pytest.mark.parametrize("k", [5, 15])
def test_k_neighbours(afm_filtered, k):
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", k=k, ncores=1)
    assert "X_umap" in afm_filtered.obsm


def test_unknown_method_raises(afm_filtered):
    with pytest.raises((ValueError, KeyError, UnboundLocalError)):
        mt.pp.reduce_dimensions(afm_filtered, method="not_a_method", ncores=1)


def test_is_deterministic_given_a_seed(afm_filtered):
    a = afm_filtered
    b = afm_filtered.copy()
    mt.pp.reduce_dimensions(a, method="PCA", seed=0, ncores=1)
    mt.pp.reduce_dimensions(b, method="PCA", seed=0, ncores=1)
    assert np.allclose(a.obsm["X_pca"], b.obsm["X_pca"])


def test_does_not_disturb_the_matrix(afm_filtered):
    shape, n_layers = afm_filtered.shape, len(afm_filtered.layers)
    mt.pp.reduce_dimensions(afm_filtered, method="PCA", ncores=1)
    assert afm_filtered.shape == shape
    assert len(afm_filtered.layers) == n_layers

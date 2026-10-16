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


# -- disconnected graphs ----------------------------------------------------
# UMAP falls back to a multi-component spectral layout when the kNN graph is
# disconnected, and forwards `metric` to sklearn.pairwise_distances there.
# MiTo's own metric names are unknown to sklearn, which used to raise.

def _disconnected_afm(**overrides):
    from conftest import build_afm
    cfg = {"n_cells": 60, "n_vars": 24, "n_clones": 6, "clone_specific_frac": 1.0, "seed": 101}
    cfg.update(overrides)
    a = build_afm(**cfg)
    mt.pp.annotate_vars(a)
    return mt.pp.filter_afm(a, filtering="baseline", compute_enrichment=False, ncores=1)


def test_fixture_really_is_disconnected():
    """Guard: if this graph ever becomes connected, the tests below stop testing anything."""
    from scipy.sparse.csgraph import connected_components
    afm = _disconnected_afm()
    _, _, conn = mt.pp.kNN_graph(D=afm.obsp["distances"].toarray(), k=5, from_distances=True)
    n_components, _ = connected_components(conn, directed=False)
    assert n_components > 1


@pytest.mark.parametrize("metric", ["weighted_jaccard", "weighted_hamming"])
def test_umap_with_custom_metric_on_disconnected_graph(metric):
    """Regression: MiTo's custom metrics used to reach sklearn and raise."""
    afm = _disconnected_afm()
    if metric == "weighted_hamming":
        pytest.skip("weighted_hamming needs per-character priors in .varm")
    mt.pp.reduce_dimensions(afm, method="UMAP", metric=metric, k=5, ncores=1)
    assert "X_umap" in afm.obsm
    assert np.isfinite(afm.obsm["X_umap"]).all()


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "jaccard"])
def test_umap_with_sklearn_metric_on_disconnected_graph(metric):
    """Metrics sklearn knows are passed through unchanged."""
    afm = _disconnected_afm()
    mt.pp.reduce_dimensions(afm, method="UMAP", metric=metric, k=5, ncores=1)
    assert "X_umap" in afm.obsm


def test_umap_metric_fallback_is_only_for_unknown_metrics():
    from mito.pp.dimred import _UMAP_SAFE_METRICS
    assert "euclidean" in _UMAP_SAFE_METRICS
    assert "jaccard" in _UMAP_SAFE_METRICS
    assert "weighted_jaccard" not in _UMAP_SAFE_METRICS
    assert "weighted_hamming" not in _UMAP_SAFE_METRICS

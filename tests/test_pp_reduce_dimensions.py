"""
mito.pp.reduce_dimensions

Embeddings have to be reproducible: a figure regenerated on another day, or on another
machine, must be the same figure. For UMAP that depends on the layout optimisation
running single-threaded, which the implementation pins explicitly.
"""

import numpy as np
import pytest

import mito as mt

METHODS = ["PCA", "UMAP", "diffmap"]
KEYS = {"PCA": "X_pca", "UMAP": "X_umap", "diffmap": "X_diffmap"}


@pytest.mark.parametrize("method", METHODS)
def test_writes_the_embedding_with_the_scanpy_key(afm_filtered, method):
    mt.pp.reduce_dimensions(afm_filtered, method=method, n_comps=2, ncores=1)
    X = afm_filtered.obsm[KEYS[method]]
    assert X.shape == (afm_filtered.shape[0], 2)
    assert np.isfinite(X).all()


@pytest.mark.parametrize("n_comps", [2, 3])
def test_n_comps_is_respected(afm_filtered, n_comps):
    mt.pp.reduce_dimensions(afm_filtered, method="PCA", n_comps=n_comps, ncores=1)
    assert afm_filtered.obsm["X_pca"].shape[1] == n_comps


@pytest.mark.parametrize("method", METHODS)
def test_the_same_seed_gives_the_same_embedding(afm_filtered, method):
    first = mt.pp.reduce_dimensions(afm_filtered, method=method, seed=42, ncores=1, copy=True)
    second = mt.pp.reduce_dimensions(afm_filtered, method=method, seed=42, ncores=1, copy=True)
    assert np.allclose(first.obsm[KEYS[method]], second.obsm[KEYS[method]])


def test_umap_is_reproducible_across_objects(afm_filtered):
    """
    The layout must depend on the data and the seed only. If the numba optimisation ran
    in parallel, two runs of the same seed would differ by a little - which is exactly
    the kind of irreproducibility that is never noticed until a figure changes.
    """
    a = afm_filtered.copy()
    b = afm_filtered.copy()
    mt.pp.reduce_dimensions(a, method="UMAP", seed=7, ncores=1)
    mt.pp.reduce_dimensions(b, method="UMAP", seed=7, ncores=1)
    assert np.array_equal(a.obsm["X_umap"], b.obsm["X_umap"])


def test_a_different_seed_gives_a_different_umap(afm_filtered):
    a = mt.pp.reduce_dimensions(afm_filtered, method="UMAP", seed=1, ncores=1, copy=True)
    b = mt.pp.reduce_dimensions(afm_filtered, method="UMAP", seed=2, ncores=1, copy=True)
    assert not np.allclose(a.obsm["X_umap"], b.obsm["X_umap"])


def test_precomputed_distances_are_reused(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray().copy()
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", metric="weighted_jaccard", ncores=1)
    assert np.allclose(afm_filtered.obsp["distances"].toarray(), D)


def test_distances_are_computed_when_the_metric_differs(afm_filtered):
    """A different metric is a different graph: it must not silently reuse the old one."""
    D = afm_filtered.obsp["distances"].toarray().copy()
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", metric="jaccard", ncores=1)
    assert not np.allclose(afm_filtered.obsp["distances"].toarray(), D)
    assert afm_filtered.uns["distances"]["distances"]["metric"] == "jaccard"


def test_records_what_it_did(afm_filtered):
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", n_comps=2, k=7, seed=3, ncores=1)
    record = afm_filtered.uns["mito"]["reduce_dimensions"]
    assert record == {"method": "UMAP", "n_comps": 2, "k": 7, "layer": "bin",
                      "metric": "weighted_jaccard", "seed": 3}


def test_unknown_method_lists_the_alternatives(afm_filtered):
    with pytest.raises(ValueError, match="PCA"):
        mt.pp.reduce_dimensions(afm_filtered, method="tSNE", ncores=1)


def test_copy_semantics(afm_filtered):
    out = mt.pp.reduce_dimensions(afm_filtered, method="PCA", ncores=1, copy=True)
    assert out is not afm_filtered
    assert "X_pca" in out.obsm and "X_pca" not in afm_filtered.obsm


def test_in_place_returns_none(afm_filtered):
    assert mt.pp.reduce_dimensions(afm_filtered, method="PCA", ncores=1) is None

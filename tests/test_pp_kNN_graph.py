"""
mito.pp.kNN_graph

Light coverage: both entry points (feature matrix and precomputed distances),
the k parameter, and the shape contract of the returned triple.
"""

import numpy as np
import pytest

import mito as mt


def test_from_distances_returns_triple(distance_matrix):
    out = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    assert isinstance(out, tuple) and len(out) == 3


def test_from_distances_index_shape(distance_matrix):
    idx, dists, conn = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    assert idx.shape == (distance_matrix.shape[0], 5)


@pytest.mark.parametrize("k", [3, 5, 10, 15])
def test_k_controls_neighbour_count(distance_matrix, k):
    """Regression: k used to be ignored, returning all n-1 neighbours."""
    idx, _, _ = mt.pp.kNN_graph(D=distance_matrix, k=k, from_distances=True)
    assert idx.shape[1] == k


def test_self_is_excluded(distance_matrix):
    idx, _, _ = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    for i in range(distance_matrix.shape[0]):
        assert i not in idx[i, :]


def test_neighbours_are_the_nearest_ones(distance_matrix):
    """The returned neighbours must be the k smallest distances for each row."""
    k = 5
    idx, _, _ = mt.pp.kNN_graph(D=distance_matrix, k=k, from_distances=True)
    for i in range(distance_matrix.shape[0]):
        expected = set(np.argsort(distance_matrix[i, :])[1:k + 1])
        assert set(idx[i, :]) == expected


def test_indices_are_in_range(distance_matrix):
    idx, _, _ = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    n = distance_matrix.shape[0]
    assert idx.min() >= 0 and idx.max() < n


def test_from_feature_matrix():
    rng = np.random.default_rng(1)
    X = rng.random((40, 6))
    idx, dists, conn = mt.pp.kNN_graph(X=X, k=5)
    assert idx.shape[0] == 40


def test_distances_matrix_is_sparse_and_non_negative(distance_matrix):
    _, dists, _ = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    n = distance_matrix.shape[0]
    assert dists.shape == (n, n)
    assert (dists.toarray() >= 0).all()


def test_connectivities_shape(distance_matrix):
    _, _, conn = mt.pp.kNN_graph(D=distance_matrix, k=5, from_distances=True)
    n = distance_matrix.shape[0]
    assert conn.shape == (n, n)


def test_on_pipeline_distances(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray()
    idx, _, _ = mt.pp.kNN_graph(D=D, k=5, from_distances=True)
    assert idx.shape[0] == afm_filtered.shape[0]


def test_k_larger_than_sample_count_raises(distance_matrix):
    with pytest.raises(ValueError, match="not smaller than the number of observations"):
        mt.pp.kNN_graph(D=distance_matrix, k=distance_matrix.shape[0] + 10,
                        from_distances=True)


def test_k_equal_to_sample_count_raises(distance_matrix):
    with pytest.raises(ValueError):
        mt.pp.kNN_graph(D=distance_matrix, k=distance_matrix.shape[0],
                        from_distances=True)

"""
mito.pp.kNN_graph

AnnData-first, like ``sc.pp.neighbors``: the graph is written onto the object, so what
computed it and with which parameters is recoverable from the object alone.
"""

import numpy as np
import pytest

import mito as mt


def test_writes_the_graph_onto_the_object(afm_filtered):
    mt.pp.kNN_graph(afm_filtered, k=10)

    assert {"neighbours_distances", "neighbours_connectivities"} <= set(afm_filtered.obsp)
    record = afm_filtered.uns["neighbours"]
    assert record["params"]["k"] == 10
    assert record["params"]["metric"] == "weighted_jaccard"
    assert record["distances_key"] == "neighbours_distances"
    assert record["connectivities_key"] == "neighbours_connectivities"


def test_no_cell_has_more_than_k_neighbours(afm_filtered):
    """
    k is an upper bound on the stored distances, not an exact count: cells with an
    identical genotype profile are at distance 0 from each other, and those entries are
    dropped from the sparse matrix. The connectivities are the graph to reason about.
    """
    k = 8
    mt.pp.kNN_graph(afm_filtered, k=k)

    stored = np.diff(afm_filtered.obsp["neighbours_distances"].indptr)
    assert stored.max() <= k

    neighbours = np.diff(afm_filtered.obsp["neighbours_connectivities"].indptr)
    assert (neighbours > 0).all(), "every cell must be connected to something"


def test_connectivities_are_symmetric_and_non_negative(afm_filtered):
    mt.pp.kNN_graph(afm_filtered, k=10)
    C = afm_filtered.obsp["neighbours_connectivities"].toarray()
    assert (C >= 0).all()
    assert np.allclose(C, C.T, atol=1e-8)


def test_key_added_lets_several_graphs_coexist(afm_filtered):
    mt.pp.kNN_graph(afm_filtered, k=5, key_added="knn5")
    mt.pp.kNN_graph(afm_filtered, k=15, key_added="knn15")
    assert afm_filtered.uns["knn5"]["params"]["k"] == 5
    assert afm_filtered.uns["knn15"]["params"]["k"] == 15
    assert {"knn5_distances", "knn15_distances"} <= set(afm_filtered.obsp)


def test_needs_distances_and_says_where_they_come_from(afm):
    with pytest.raises(ValueError, match="compute_distances"):
        mt.pp.kNN_graph(afm)


def test_k_larger_than_the_dataset_is_refused(afm_filtered):
    with pytest.raises(ValueError, match="observations"):
        mt.pp.kNN_graph(afm_filtered, k=afm_filtered.shape[0] + 5)


def test_copy_semantics(afm_filtered):
    out = mt.pp.kNN_graph(afm_filtered, k=10, copy=True)
    assert out is not afm_filtered
    assert "neighbours" in out.uns and "neighbours" not in afm_filtered.uns


def test_in_place_returns_none(afm_filtered):
    assert mt.pp.kNN_graph(afm_filtered, k=10) is None

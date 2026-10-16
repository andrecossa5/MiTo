"""
mito.tl.build_tree
"""

import numpy as np
import pytest

import mito as mt

SOLVERS = ["UPMGA", "NJ", "spectral", "greedy"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_returns_a_tree_over_the_cells(afm_filtered, solver):
    tree = mt.tl.build_tree(afm_filtered, solver=solver)
    assert set(tree.leaves) == set(afm_filtered.obs_names)
    assert len(tree.internal_nodes) > 0


def test_cell_metadata_travels_with_the_tree(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered)
    assert "GBC" in tree.cell_meta.columns
    assert list(tree.cell_meta.index) == list(afm_filtered.obs_names)


def test_layers_carry_the_character_matrices(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered)
    assert set(tree.layers) >= {"raw", "transformed"}
    assert tree.layers["raw"].shape[0] == afm_filtered.shape[0]
    assert set(np.unique(tree.layers["transformed"].values)) <= {0, 1, -1}


def test_precomputed_distances_are_reused(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray().copy()
    mt.tl.build_tree(afm_filtered)
    assert np.allclose(afm_filtered.obsp["distances"].toarray(), D)


def test_distances_are_computed_when_absent(afm_filtered):
    del afm_filtered.obsp["distances"]
    del afm_filtered.uns["distances"]
    tree = mt.tl.build_tree(afm_filtered, metric="weighted_jaccard", ncores=1)
    assert "distances" in afm_filtered.obsp
    assert set(tree.leaves) == set(afm_filtered.obs_names)


def test_recovers_the_planted_clones(afm_filtered):
    """Cells of one clone must be more similar to each other than to the rest."""
    tree = mt.tl.build_tree(afm_filtered)
    truth = afm_filtered.obs["GBC"].astype(str)
    D = afm_filtered.obsp["distances"].toarray()
    for clone in truth.unique():
        inside = (truth == clone).values
        within = D[np.ix_(inside, inside)].mean()
        between = D[np.ix_(inside, ~inside)].mean()
        assert within < between


def test_solver_is_validated(afm_filtered):
    with pytest.raises(KeyError):
        mt.tl.build_tree(afm_filtered, solver="not_a_solver")


def test_is_deterministic(afm_filtered):
    a = mt.tl.build_tree(afm_filtered.copy(), solver="UPMGA")
    b = mt.tl.build_tree(afm_filtered.copy(), solver="UPMGA")
    assert a.get_newick() == b.get_newick()


def test_filter_muts_drops_uninformative_characters(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered, filter_muts=True, min_n_positive_cells=5)
    assert tree.layers["transformed"].shape[1] <= afm_filtered.shape[1]


def _clades(tree):
    return {frozenset(tree.leaves_in_subtree(n)) for n in tree.internal_nodes}


def test_collapsing_ties_keeps_the_upgma_clades(afm_filtered):
    binary = mt.tl.build_tree(afm_filtered.copy(), collapse_ties=False)
    collapsed = mt.tl.build_tree(afm_filtered.copy())
    assert set(collapsed.leaves) == set(binary.leaves)
    assert _clades(collapsed) <= _clades(binary)
    assert len(collapsed.internal_nodes) <= len(binary.internal_nodes)


def test_collapsed_tree_has_upgma_branch_lengths(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered)
    lengths = [tree.get_branch_length(u, v) for u, v in tree.edges]
    assert min(lengths) >= 0 and max(lengths) > 0
    # ultrametric: every leaf at the same distance from the root
    depths = [tree.get_time(leaf) for leaf in tree.leaves]
    assert np.allclose(depths, depths[0])


def test_clones_do_not_depend_on_collapsing_ties(afm_filtered):
    labels = []
    for collapse in (False, True):
        afm = afm_filtered.copy()
        mt.tl.annotate_clones(mt.tl.build_tree(afm, collapse_ties=collapse), afm, support=False)
        labels.append(afm.obs["MiTo_clone"].astype(str))
    assert (labels[0] == labels[1]).all()

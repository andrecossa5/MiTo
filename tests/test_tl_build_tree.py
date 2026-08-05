"""
mito.tl.build_tree

Light coverage: the solvers MiTo exposes, the precomputed-distance path, and the
tree's structural contract against the AFM it came from.
"""

import pytest

import mito as mt

# The solvers MiTo exposes. shared_muts and max_cut were removed; NJ works from
# cassiopeia-mt 2.1.2, which fixed the dissimilarity-map dtype under pandas 3.
SOLVERS = ["NJ", "UPMGA", "spectral", "greedy"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_solver_builds_a_tree(afm_filtered, solver):
    tree = mt.tl.build_tree(afm_filtered, precomputed=True, solver=solver)
    assert hasattr(tree, "leaves")
    assert len(tree.leaves) == afm_filtered.shape[0]


@pytest.mark.parametrize("solver", SOLVERS)
def test_leaves_match_cell_names(afm_filtered, solver):
    tree = mt.tl.build_tree(afm_filtered, precomputed=True, solver=solver)
    assert set(tree.leaves) == set(afm_filtered.obs_names)


def test_tree_has_internal_structure(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered, precomputed=True, solver="UPMGA")
    assert len(tree.internal_nodes) > 0
    assert tree.root is not None


def test_character_matrix_matches_afm(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered, precomputed=True, solver="UPMGA")
    assert tree.character_matrix.shape[0] == afm_filtered.shape[0]


def test_precomputed_false_recomputes_distances(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered, precomputed=False, solver="UPMGA")
    assert len(tree.leaves) == afm_filtered.shape[0]


@pytest.mark.parametrize("metric", ["weighted_jaccard", "jaccard"])
def test_metric_choice(afm_filtered, metric):
    tree = mt.tl.build_tree(afm_filtered, precomputed=False, metric=metric,
                            solver="UPMGA")
    assert len(tree.leaves) == afm_filtered.shape[0]


@pytest.mark.parametrize("bin_method", ["MiTo", "vanilla"])
def test_bin_method_choice(afm_filtered, bin_method):
    tree = mt.tl.build_tree(afm_filtered, precomputed=False, bin_method=bin_method,
                            solver="UPMGA")
    assert len(tree.leaves) == afm_filtered.shape[0]


def test_unknown_solver_raises(afm_filtered):
    with pytest.raises((KeyError, ValueError)):
        mt.tl.build_tree(afm_filtered, precomputed=True, solver="not_a_solver")


def test_filter_muts_toggle(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered, precomputed=True, solver="UPMGA",
                            filter_muts=True)
    assert len(tree.leaves) <= afm_filtered.shape[0]


def test_is_deterministic(afm_filtered):
    a = mt.tl.build_tree(afm_filtered.copy(), precomputed=True, solver="UPMGA")
    b = mt.tl.build_tree(afm_filtered.copy(), precomputed=True, solver="UPMGA")
    assert set(a.leaves) == set(b.leaves)


@pytest.mark.parametrize("solver", ["shared_muts", "max_cut"])
def test_removed_solvers_are_rejected(afm_filtered, solver):
    """These two were dropped from MiTo's solver registry."""
    with pytest.raises((KeyError, ValueError)):
        mt.tl.build_tree(afm_filtered, precomputed=True, solver=solver)


def test_solver_registry_matches_expectation():
    """Guard: the exposed solver set is exactly the supported one."""
    from mito.tl.phylo import _solver_kwargs, solver_d
    assert set(solver_d) == set(SOLVERS)
    assert set(_solver_kwargs) == set(SOLVERS)

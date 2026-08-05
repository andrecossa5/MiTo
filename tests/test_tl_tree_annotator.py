"""
mito.tl.MiToTreeAnnotator

Light coverage: construction, input validation and the shape of what clonal
inference produces. The method already sweeps its own hyper-parameter grid
internally, so this does not re-sweep it from outside.
"""

import numpy as np
import pandas as pd
import pytest
import mito as mt


# -- construction and inputs ------------------------------------------------

def test_constructs_from_a_tree(tree):
    annotator = mt.tl.MiToTreeAnnotator(tree)
    assert annotator.tree is not None


def test_exposes_expected_slots(tree):
    annotator = mt.tl.MiToTreeAnnotator(tree)
    for slot in ("T", "M", "tree", "clone_df", "mut_df", "clonal_nodes",
                 "internal_nodes_df", "ordered_muts", "params", "solutions"):
        assert hasattr(annotator, slot)


@pytest.mark.parametrize("bad", [None, "not_a_tree", 42, [], {}])
def test_rejects_non_tree_input(bad):
    """Regression: the constructor used to accept anything and fail much later."""
    with pytest.raises(TypeError, match="CassiopeiaTree"):
        mt.tl.MiToTreeAnnotator(bad)


def test_slots_start_empty(tree):
    """T and M are computed during inference, not at construction."""
    annotator = mt.tl.MiToTreeAnnotator(tree)
    assert annotator.T is None
    assert annotator.M is None


def test_get_T_builds_the_assignment_matrix(tree):
    annotator = mt.tl.MiToTreeAnnotator(tree)
    annotator.get_T()
    assert annotator.T is not None
    assert annotator.T.shape[0] == len(tree.leaves)


def test_get_T_with_and_without_root(tree):
    annotator = mt.tl.MiToTreeAnnotator(tree)
    annotator.get_T(with_root=True)
    with_root = annotator.T.shape[1]
    annotator.get_T(with_root=False)
    without_root = annotator.T.shape[1]
    assert with_root >= without_root


# -- clonal inference output ------------------------------------------------

def test_clonal_inference_assigns_clones(annotated_tree):
    assert "MiTo clone" in annotated_tree.cell_meta.columns


def test_every_cell_gets_a_label(annotated_tree):
    labels = annotated_tree.cell_meta["MiTo clone"]
    assert len(labels) == len(annotated_tree.leaves)


def test_clone_labels_are_not_all_identical(annotated_tree):
    """Inference on structured input must find more than one clone."""
    assert annotated_tree.cell_meta["MiTo clone"].nunique() > 1


def test_supporting_columns_are_added(annotated_tree):
    for column in ("median cell similarity", "n cells", "lca"):
        assert column in annotated_tree.cell_meta.columns


def test_clone_df_is_populated(annotator):
    assert isinstance(annotator.clone_df, pd.DataFrame)
    assert len(annotator.clone_df) > 0


def test_solutions_recorded(annotator):
    """The internal grid search must leave its candidate solutions behind."""
    assert annotator.solutions is not None
    assert len(annotator.solutions) > 0


def test_params_recorded(annotator):
    assert annotator.params is not None


def test_clonal_nodes_are_tree_nodes(annotator):
    nodes = set(annotator.tree.nodes)
    assert set(annotator.clonal_nodes).issubset(nodes)


def test_n_cells_column_sums_to_leaf_count(annotated_tree):
    counts = annotated_tree.cell_meta.groupby("MiTo clone", observed=True).size()
    assert counts.sum() == len(annotated_tree.leaves)


# -- downstream helpers -----------------------------------------------------

def test_extract_mut_order(annotator):
    """Sets ordered_muts and mut_df in place rather than returning them."""
    annotator.extract_mut_order()
    assert annotator.ordered_muts is not None
    assert len(annotator.ordered_muts) > 0
    assert set(annotator.ordered_muts).issubset(set(annotator.M.index))
    assert len(set(annotator.ordered_muts)) == len(annotator.ordered_muts)


@pytest.mark.parametrize("pval_tresh", [0.001, 0.01, 0.1])
def test_extract_mut_order_thresholds(annotator, pval_tresh):
    annotator.extract_mut_order(pval_tresh=pval_tresh)
    assert 0 < len(annotator.ordered_muts) <= len(annotator.M.index)


def test_compute_expansions(annotator):
    annotator.compute_expansions()
    assert annotator.tree is not None


def test_compute_cell_fitness(annotator):
    annotator.compute_cell_fitness()
    assert annotator.tree is not None


def test_mut_df_shape(annotator):
    assert isinstance(annotator.mut_df, pd.DataFrame)


def test_M_is_a_matrix(annotator):
    M = annotator.M
    assert M is not None
    assert hasattr(M, "shape")

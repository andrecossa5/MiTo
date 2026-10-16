"""
mito.tl.collapse_clones
"""

import numpy as np
import pytest

import mito as mt


def _clone_labels(tree):
    return tree.cell_meta["MiTo_clone"].astype(str)


def test_one_tip_per_clone_with_the_cells_of_the_clone(annotated_tree):
    flat = mt.tl.collapse_clones(annotated_tree)
    labels = _clone_labels(annotated_tree)
    assigned = labels[labels != "unassigned"]
    assert set(flat.cell_meta["clone"]) == set(assigned.unique())
    assert flat.cell_meta["n_cells"].sum() == len(assigned)
    for tip, row in flat.cell_meta.iterrows():
        assert set(labels.loc[list(row["cells"])]) == {row["clone"]}


def test_branch_lengths_keep_the_tree_ultrametric(annotated_tree):
    flat = mt.tl.collapse_clones(annotated_tree)
    depths = [flat.get_time(leaf) for leaf in flat.leaves]
    assert np.allclose(depths, depths[0])
    assert np.isclose(depths[0], max(annotated_tree.get_time(leaf) for leaf in annotated_tree.leaves))


def test_clone_support_is_carried_to_the_tip(annotated_tree):
    flat = mt.tl.collapse_clones(annotated_tree)
    for tip, row in flat.cell_meta.iterrows():
        cells = list(row["cells"])
        if len(cells) > 1:
            node = annotated_tree.find_lca(*cells)
            expected = annotated_tree.get_attribute(node, "support")
            assert np.isclose(row["support"], expected)
            assert np.isclose(flat.get_attribute(tip, "support"), expected)


def test_unassigned_cells_can_be_kept(annotated_tree):
    flat = mt.tl.collapse_clones(annotated_tree, show_unassigned=True)
    n_unassigned = (_clone_labels(annotated_tree) == "unassigned").sum()
    assert flat.cell_meta["clone"].isna().sum() == n_unassigned


def test_needs_the_clone_labels(afm_filtered):
    tree = mt.tl.build_tree(afm_filtered)
    with pytest.raises(KeyError, match="annotate_clones"):
        mt.tl.collapse_clones(tree)


def test_plot_clone_tree_accepts_a_collapsed_tree_colored_by_support(annotated_tree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flat = mt.tl.collapse_clones(annotated_tree)
    mt.pl.plot_clone_tree(flat, color_by="support")
    plt.close("all")


def test_every_leaf_of_the_clone_tree_is_a_clone(annotated_tree):
    flat = mt.tl.collapse_clones(annotated_tree)
    assert set(flat.leaves) == set(flat.cell_meta.index)

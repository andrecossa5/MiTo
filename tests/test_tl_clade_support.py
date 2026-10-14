"""
mito.tl.clade_support, and its use in mito.tl.annotate_clones

Support is how often a clade is rebuilt when the molecule counts are drawn again. Planted
clones carried by strong markers must be supported; the clades UPGMA forms among cells
with no shared marker must not.
"""

import numpy as np
import pytest

import mito as mt


def test_planted_clones_are_supported_and_noise_clades_are_not(annotated):
    afm, tree = annotated
    sup = mt.tl.clade_support(tree, afm, n_boot=30)
    assert set(sup.index) == set(tree.internal_nodes)
    assert sup.between(0, 1).all() and sup[tree.root] == 1
    nodes = afm.uns['mito']['annotate_clones']['clones']['node']
    assert sup[nodes].min() >= 0.8                      # every called clone is robust
    others = [x for x in tree.internal_nodes if x not in set(nodes) and x != tree.root]
    assert sup[others].median() < sup[nodes].median()


def test_support_is_written_on_the_tree_and_recorded(annotated):
    afm, tree = annotated
    sup = mt.tl.clade_support(tree, afm, n_boot=10, seed=3)
    node = sup.index[0]
    assert tree.get_attribute(node, 'support') == pytest.approx(sup[node])
    assert afm.uns['mito']['clade_support']['params']['n_boot'] == 10


def test_exact_support_is_never_above_transfer_support(annotated):
    afm, tree = annotated
    t = mt.tl.clade_support(tree, afm, n_boot=20, seed=1)
    e = mt.tl.clade_support(tree, afm, n_boot=20, seed=1, method='exact', key_added='exact')
    assert (e <= t + 1e-9).all()


def test_is_reproducible(annotated):
    afm, tree = annotated
    a = mt.tl.clade_support(tree, afm, n_boot=10, seed=7)
    b = mt.tl.clade_support(tree, afm, n_boot=10, seed=7)
    assert np.allclose(a, b)


def test_refuses_a_tree_of_other_cells(annotated):
    afm, tree = annotated
    with pytest.raises(ValueError, match='leaves'):
        mt.tl.clade_support(tree, afm[:-1].copy(), n_boot=2)


def test_annotate_clones_reports_and_filters_on_support(annotated):
    afm, tree = annotated
    mt.tl.clade_support(tree, afm, n_boot=20)
    mt.tl.annotate_clones(tree, afm)
    clones = afm.uns['mito']['annotate_clones']['clones']
    assert 'support' in clones.columns and clones['support'].notna().all()
    before = (afm.obs['MiTo_clone'] != 'unassigned').sum()
    mt.tl.annotate_clones(tree, afm, min_support=1.01)
    assert (afm.obs['MiTo_clone'] == 'unassigned').all()
    mt.tl.annotate_clones(tree, afm, min_support=0.0)
    assert (afm.obs['MiTo_clone'] != 'unassigned').sum() == before


def test_annotate_clones_computes_support_when_the_tree_has_none(afm_filtered, tree):
    mt.tl.annotate_clones(tree, afm_filtered, n_boot=10)
    assert 'clade_support' in afm_filtered.uns['mito']
    assert afm_filtered.uns['mito']['annotate_clones']['clones']['support'].notna().all()


def test_min_support_needs_support(afm_filtered, tree):
    with pytest.raises(ValueError, match='clade_support'):
        mt.tl.annotate_clones(tree, afm_filtered, support=False, min_support=0.5)

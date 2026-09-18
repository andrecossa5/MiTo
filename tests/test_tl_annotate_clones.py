"""
mito.tl.annotate_clones, evidence_cut, clone_support, rescue_unassigned

The cut splits a clade only when its children carry markers specific against their
siblings; a cell then keeps its label only if its own call is corroborated by its
neighbourhood. Both halves are tested here, on planted clones whose answer is known.
"""

import numpy as np
import pandas as pd
import pytest

import mito as mt

# -- the labels -------------------------------------------------------------


def test_recovers_the_planted_clones(annotated):
    afm, _ = annotated
    labels = afm.obs["MiTo_clone"].astype(str)
    assigned = labels != "unassigned"
    truth = afm.obs["GBC"].astype(str)
    assert mt.ut.custom_ARI(truth[assigned], labels[assigned]) > 0.9


def test_labels_are_stable_names_ordered_by_size(annotated):
    afm, _ = annotated
    counts = afm.obs["MiTo_clone"].astype(str).value_counts()
    clones = [c for c in counts.index if c != "unassigned"]
    assert all(c.startswith("MT-") for c in clones)
    # MT-1 is the largest, MT-2 the next, and so on
    sizes = [counts[f"MT-{i}"] for i in range(1, len(clones) + 1)]
    assert sizes == sorted(sizes, reverse=True)


def test_writes_the_labels_on_both_the_afm_and_the_tree(annotated):
    afm, tree = annotated
    assert {"MiTo_clone", "clone_support"} <= set(afm.obs.columns)
    assert {"MiTo_clone", "clone_support"} <= set(tree.cell_meta.columns)
    assert (tree.cell_meta["MiTo_clone"].values ==
            afm.obs.loc[tree.cell_meta.index, "MiTo_clone"].astype(str).values).all()


def test_the_clone_table_says_where_each_clone_came_from(annotated):
    afm, tree = annotated
    clones = afm.uns["mito"]["annotate_clones"]["clones"]
    assert list(clones.columns) == ["n_cells", "node", "markers"]
    assert clones["n_cells"].sum() == (afm.obs["MiTo_clone"].astype(str) != "unassigned").sum()
    assert set(clones["node"]) <= set(tree.nodes)
    assert (clones["markers"].str.len() > 0).all(), "a clone with no marker is not a clone"


def test_records_its_parameters(annotated):
    afm, _ = annotated
    record = afm.uns["mito"]["annotate_clones"]
    assert record["params"]["tau"] == 0.25
    assert 0 < record["frac_assigned"] <= 1


# -- abstention -------------------------------------------------------------


def test_abstention_only_removes_labels(afm_filtered, tree):
    strict = afm_filtered.copy()
    lenient = afm_filtered.copy()
    import copy
    mt.tl.annotate_clones(copy.deepcopy(tree), strict, tau=0.9)
    mt.tl.annotate_clones(copy.deepcopy(tree), lenient, tau=0.0)

    n_strict = (strict.obs["MiTo_clone"].astype(str) != "unassigned").sum()
    n_lenient = (lenient.obs["MiTo_clone"].astype(str) != "unassigned").sum()
    assert n_strict <= n_lenient


def test_clone_support_is_a_neighbourhood_fraction(annotated):
    afm, _ = annotated
    support = afm.obs["clone_support"].values
    assert ((support >= 0) & (support <= 1)).all()
    assigned = (afm.obs["MiTo_clone"].astype(str) != "unassigned").values
    assert support[assigned].min() >= 0.25


def test_clone_support_matrix_matches_the_calls(afm_filtered):
    X = afm_filtered.X.toarray()
    B = afm_filtered.layers["bin"].toarray() > 0
    support = mt.tl.clone_support(X, B, k=10)
    assert support.shape == B.shape
    assert ((support >= 0) & (support <= 1)).all()


# -- the cut itself ---------------------------------------------------------


def test_evidence_cut_labels_every_cell(afm_filtered, tree):
    B = afm_filtered.layers["bin"].toarray() > 0
    labels = mt.tl.evidence_cut(tree, B.astype(float), list(afm_filtered.obs_names))
    assert list(labels.index) == list(afm_filtered.obs_names)
    assert (labels != "unassigned").all(), "the cut assigns every cell; abstention comes after"


def test_a_marked_clade_ends_up_in_one_label(afm_filtered, tree):
    """
    A single character carried by one clade: its carriers must be collected into one
    label, and that label must be (almost) pure.

    NB: the cut keeps descending through nodes it finds no evidence for, so a
    deliberately impoverished character set yields many labels below the marked clade.
    That is by design - resolving that is the abstention step's job, not the cut's - so
    what is asserted here is the placement of the carriers, not the label count.
    """
    truth = afm_filtered.obs["GBC"].astype(str)
    one_clone = (truth == truth.unique()[0]).values
    B = one_clone.astype(float).reshape(-1, 1)

    labels = mt.tl.evidence_cut(tree, B, list(afm_filtered.obs_names), min_cells=5)
    top = labels[one_clone].value_counts(normalize=True)
    assert top.iloc[0] > 0.8, "the carriers must share one label"
    winner = top.index[0]
    assert one_clone[(labels == winner).values].mean() > 0.8, "that label must be pure"


def test_without_any_character_nothing_is_assigned(afm_filtered, tree):
    """
    The cut labels clades whatever the evidence; it is the abstention step that refuses
    to call cells whose genotypes support nothing. With no calls at all, every cell must
    end up unassigned.
    """
    afm = afm_filtered.copy()
    afm.layers["bin"] = type(afm.layers["bin"])(np.zeros(afm.shape, dtype=np.int8))
    mt.tl.annotate_clones(tree, afm)
    assert (afm.obs["MiTo_clone"].astype(str) == "unassigned").all()


def test_min_supported_controls_how_readily_a_clade_is_split(afm_filtered, tree):
    B = (afm_filtered.layers["bin"].toarray() > 0).astype(float)
    cells = list(afm_filtered.obs_names)
    permissive = mt.tl.evidence_cut(tree, B, cells, min_supported=1, one_sided=True)
    strict = mt.tl.evidence_cut(tree, B, cells, min_supported=2)
    assert permissive.nunique() >= strict.nunique()


# -- rescue -----------------------------------------------------------------


def test_rescue_can_only_assign_unassigned_cells(afm_filtered, tree):
    afm = afm_filtered.copy()
    mt.tl.annotate_clones(tree, afm, tau=0.9)          # abstain aggressively
    before = afm.obs["MiTo_clone"].astype(str)
    if (before == "unassigned").sum() == 0:
        pytest.skip("nothing was left unassigned")

    labels = mt.tl.rescue_unassigned(before.copy(), afm)
    assigned_before = before != "unassigned"
    assert (labels[assigned_before] == before[assigned_before]).all()
    assert (labels != "unassigned").sum() >= assigned_before.sum()


def test_rescue_through_annotate_clones(afm_filtered, tree):
    afm = afm_filtered.copy()
    mt.tl.annotate_clones(tree, afm, rescue=True)
    assert afm.uns["mito"]["annotate_clones"]["params"]["rescue"] is True


# -- refusals and copy ------------------------------------------------------


def test_needs_genotypes(afm, tree):
    with pytest.raises(ValueError, match="filter_afm"):
        mt.tl.annotate_clones(tree, afm)


def test_copy_semantics(afm_filtered, tree):
    out = mt.tl.annotate_clones(tree, afm_filtered, copy=True)
    assert out is not afm_filtered
    assert "MiTo_clone" in out.obs.columns
    assert "MiTo_clone" not in afm_filtered.obs.columns


def test_key_added_is_respected(afm_filtered, tree):
    mt.tl.annotate_clones(tree, afm_filtered, key_added="clone_id")
    assert "clone_id" in afm_filtered.obs.columns
    assert "clone_id" in afm_filtered.cell_meta.columns if hasattr(afm_filtered, "cell_meta") else True
    assert "clone_id" in tree.cell_meta.columns


# -- downstream -------------------------------------------------------------


def test_clonal_fate_bias_runs_on_the_annotation(annotated):
    afm, tree = annotated
    tree.cell_meta["state"] = pd.Categorical(
        np.where(tree.cell_meta["clone_support"] > 0.9, "high", "low")
    )
    df = mt.tl.compute_clonal_fate_bias(tree, state_column="state", clone_column="MiTo_clone",
                                        target_state="high")
    assert {"perc_in_target_state", "odds_ratio", "FDR"} <= set(df.columns)
    assert len(df) == tree.cell_meta["MiTo_clone"].nunique()
